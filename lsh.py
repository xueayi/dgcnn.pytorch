import torch
import math
import numpy as np

class LSHIndex:
    def __init__(self, dim, num_tables=10, hash_size=512):
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # 生成随机投影矩阵，并量化为定点数
        self.projections = torch.randn(size=(num_tables, dim, hash_size))
        self.projections = torch.round(self.projections * 1024) / 1024  # 10位定点数
        
        # 初始化哈希表
        self.hash_tables = [dict() for _ in range(num_tables)]
        
    def _hash(self, x):
        # 定点数矩阵乘法
        x = torch.round(x * 1024) / 1024  # 输入量化
        projections = torch.matmul(x, self.projections)  # (B,N,num_tables,hash_size)
        return (projections > 0).int()  # 二值化哈希码
    
    def build(self, x):
        # 构建哈希表索引
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.transpose(2, 1).contiguous()  # (B,N,C)
        
        # 并行计算所有点的哈希值
        hashes = self._hash(x)  # (B,N,num_tables,hash_size)
        
        # 构建多表索引
        for table_idx in range(self.num_tables):
            batch_hashes = hashes[:, :, table_idx, :]  # (B,N,hash_size)
            for b in range(batch_size):
                for n in range(num_points):
                    h = tuple(batch_hashes[b, n].tolist())
                    if h not in self.hash_tables[table_idx]:
                        self.hash_tables[table_idx][h] = []
                    self.hash_tables[table_idx][h].append((b, n))
    
    def query(self, x, k):
        # LSH查询K近邻
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.transpose(2, 1).contiguous()  # (B,N,C)
        
        # 计算查询点的哈希值
        hashes = self._hash(x)  # (B,N,num_tables,hash_size)
        
        # 在每个哈希表中查找邻居
        indices = torch.zeros(batch_size, num_points, k).long().to(x.device)
        for b in range(batch_size):
            for n in range(num_points):
                # 收集所有表中的候选点
                candidates = set()
                for table_idx in range(self.num_tables):
                    h = tuple(hashes[b, n, table_idx].tolist())
                    if h in self.hash_tables[table_idx]:
                        candidates.update(self.hash_tables[table_idx][h])
                
                # 计算实际距离并选择最近的k个点
                if len(candidates) > 0:
                    candidates = [(c[0], c[1]) for c in candidates if c[0] == b]  # 只选择同一batch内的点
                    if len(candidates) > 0:
                        points = x[b, [c[1] for c in candidates]]  # (M,C)
                        query_point = x[b, n].unsqueeze(0)  # (1,C)
                        dists = torch.sum((points - query_point) ** 2, dim=1)  # (M,)
                        _, topk_indices = torch.topk(dists, min(k, len(candidates)), largest=False)
                        for ki in range(min(k, len(candidates))):
                            indices[b, n, ki] = candidates[topk_indices[ki]][1]
                
                # 如果候选点不足k个，用随机点填充
                if len(candidates) < k:
                    remaining = k - len(candidates)
                    random_indices = torch.randperm(num_points)[:remaining]
                    indices[b, n, len(candidates):] = random_indices
        
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
    lsh = LSHIndex(dim=channels, num_tables=10, hash_size=int(128*math.log2(num_points)))
    
    # 构建索引并查询
    lsh.build(x)
    indices = lsh.query(x, k)
    
    return indices