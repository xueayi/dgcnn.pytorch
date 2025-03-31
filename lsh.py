import torch
import math
import numpy as np

class LSHIndex:
    def __init__(self, dim, num_tables=10, hash_size=512, device=None):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 生成随机投影矩阵，并量化为定点数
        self.projections = torch.randn(size=(num_tables, dim, hash_size), device=self.device)
        self.projections = torch.round(self.projections * 1024) / 1024  # 10位定点数
        
        # 初始化哈希表
        self.hash_tables = [dict() for _ in range(num_tables)]
        
    def _hash(self, x):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        # 定点数矩阵乘法
        x = torch.round(x * 1024) / 1024  # 输入量化
        # 调整维度以适应批处理
        x_shape = x.shape
        x_reshaped = x.view(-1, x_shape[-1])  # (B*N, C)
        # 计算每个表的哈希值
        hashes = []
        for i in range(self.num_tables):
            # 使用当前表的投影矩阵
            proj = self.projections[i]  # (dim, hash_size)
            # 计算当前表的哈希值
            table_hash = torch.matmul(x_reshaped, proj)  # (B*N, hash_size)
            hashes.append(table_hash)
        # 将所有表的哈希值堆叠在一起
        hashes = torch.stack(hashes, dim=1)  # (B*N, num_tables, hash_size)
        hashes = hashes.view(*x_shape[:-1], self.num_tables, self.hash_size)  # (B,N,num_tables,hash_size)
        return (hashes > 0).int()  # 二值化哈希码
    
    def build(self, x):
        # 构建哈希表索引
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.transpose(2, 1).contiguous()  # (B,N,C)
        
        # 并行计算所有点的哈希值
        hashes = self._hash(x)  # (B,N,num_tables,hash_size)
        
        # 批量构建多表索引
        for table_idx in range(self.num_tables):
            batch_hashes = hashes[:, :, table_idx, :]  # (B,N,hash_size)
            # 将哈希码转换为整数以加速查找
            hash_ints = torch.packbits(batch_hashes.bool(), dim=-1)  # 压缩二进制哈希码
            hash_keys = hash_ints.cpu().numpy()
            
            # 批量更新哈希表
            for b in range(batch_size):
                # 使用numpy的unique加速重复哈希值的处理
                unique_hashes, inverse_indices = np.unique(hash_keys[b], return_inverse=True, axis=0)
                for i, h in enumerate(unique_hashes):
                    h_tuple = tuple(h.tolist())
                    points = np.where(inverse_indices == i)[0]
                    if h_tuple not in self.hash_tables[table_idx]:
                        self.hash_tables[table_idx][h_tuple] = []
                    self.hash_tables[table_idx][h_tuple].extend([(b, int(p)) for p in points])
    
    def query(self, x, k):
        # LSH查询K近邻
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.transpose(2, 1).contiguous()  # (B,N,C)
        
        # 计算查询点的哈希值
        hashes = self._hash(x)  # (B,N,num_tables,hash_size)
        hash_ints = torch.packbits(hashes.bool(), dim=-1)  # 压缩二进制哈希码
        
        # 预分配结果张量
        indices = torch.zeros(batch_size, num_points, k, dtype=torch.long, device=x.device)
        
        # 批量处理每个batch
        for b in range(batch_size):
            # 预计算当前batch中所有点对之间的距离矩阵
            points_b = x[b]  # (N,C)
            dist_matrix = torch.cdist(points_b, points_b)  # (N,N)
            
            # 并行处理当前batch中的所有点
            for n in range(num_points):
                # 快速收集所有表中的候选点
                candidates = set()
                query_hash = hash_ints[b, n].cpu().numpy()
                
                # 并行查询所有哈希表
                for table_idx in range(self.num_tables):
                    h = tuple(query_hash[table_idx].tolist())
                    if h in self.hash_tables[table_idx]:
                        # 只添加同一batch内的点
                        candidates.update(p[1] for p in self.hash_tables[table_idx][h] if p[0] == b)
                
                if candidates:
                    # 使用预计算的距离矩阵快速获取最近邻
                    candidates = torch.tensor(list(candidates), device=x.device)
                    dists = dist_matrix[n, candidates]
                    _, topk_indices = torch.topk(dists, min(k, len(candidates)), largest=False)
                    valid_neighbors = candidates[topk_indices]
                    indices[b, n, :len(valid_neighbors)] = valid_neighbors
                    
                    # 如果候选点不足k个，使用随机采样补充
                    if len(valid_neighbors) < k:
                        mask = torch.ones(num_points, dtype=torch.bool, device=x.device)
                        mask[valid_neighbors] = False
                        remaining_points = torch.nonzero(mask).squeeze(1)
                        if len(remaining_points) > 0:
                            perm = torch.randperm(len(remaining_points), device=x.device)
                            random_indices = remaining_points[perm[:k-len(valid_neighbors)]]
                            indices[b, n, len(valid_neighbors):] = random_indices
                else:
                    # 如果没有找到候选点，随机选择k个不同的点
                    perm = torch.randperm(num_points, device=x.device)
                    indices[b, n] = perm[:k]
        
        return indices

def lsh_knn(x, k):
    """LSH加速的KNN搜索，保持与原knn函数相同的接口
    Args:
        x: 输入特征 (batch_size, channels, num_points)
        k: 近邻数量
    Returns:
        idx: KNN索引 (batch_size, num_points, k)
    """
    # 初始化LSH索引
    batch_size, channels, num_points = x.size()
    device = x.device
    lsh = LSHIndex(dim=channels, num_tables=10, hash_size=int(128*math.log2(num_points)), device=device)
    
    # 构建索引并查询
    try:
        lsh.build(x)
        indices = lsh.query(x, k)
    except Exception as e:
        print(f"LSH failed: {str(e)}, fallback to brute force")
        # 暴力计算KNN
        x_t = x.transpose(2, 1).contiguous()  # (B,N,C)
        inner = -2 * torch.matmul(x_t, x)  # (B,N,N)
        xx = torch.sum(x_t ** 2, dim=2, keepdim=True)  # (B,N,1)
        pairwise_distance = xx + inner + xx.transpose(2, 1)  # (B,N,N)
        indices = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (B,N,k)
    
    return indices