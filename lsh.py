import torch
import math

class LSHIndex:
    def __init__(self, dim, num_tables=10, hash_size=512):
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # 生成随机投影矩阵
        self.projections = torch.randn(size=(num_tables, dim, hash_size))
        # FPGA优化点：此处矩阵可预量化为定点数
        
    def _hash(self, x):
        # 投影并二值化
        projections = torch.matmul(x, self.projections)  # (B,N,num_tables,hash_size)
        return (projections > 0).int()  # 二值化哈希码

    def build(self, x):
        self.hash_tables = [dict() for _ in range(self.num_tables)]
        hashes = self._hash(x)  # (B,N,num_tables,hash_size)
        
        # 构建多表索引
        for table_idx in range(self.num_tables):
            batch_hashes = hashes[:, :, table_idx, :]  # (B,N,hash_size)
            for b in range(batch_hashes.size(0)):
                for n in range(batch_hashes.size(1)):
                    hash_key = tuple(batch_hashes[b,n].cpu().numpy().tolist())
                    if hash_key not in self.hash_tables[table_idx]:
                        self.hash_tables[table_idx][hash_key] = []
                    self.hash_tables[table_idx][hash_key].append((b, n))

def lsh_knn(x, k=20, num_tables=10, hash_size=512):
    """
    LSH加速的KNN搜索
    输入：
        x : (B, C, N) 输入点云
        k : 近邻数
    输出：
        indices : (B, N, k) 近邻索引
    """
    B, C, N = x.shape
    x = x.transpose(1, 2).contiguous()  # (B, N, C)
    
    # 初始化LSH索引
    lsh_index = LSHIndex(dim=C, num_tables=num_tables, hash_size=hash_size)
    lsh_index.build(x)
    
    # 并行化查询（FPGA友好结构）
    indices = torch.zeros((B, N, k), dtype=torch.long, device=x.device)
    for b in range(B):
        for n in range(N):
            # 多表联合查询
            candidates = set()
            query_hash = lsh_index._hash(x[b:b+1, n:n+1])
            
            for table_idx in range(lsh_index.num_tables):
                hash_key = tuple(query_hash[0,0,table_idx].cpu().numpy().tolist())
                candidates.update(lsh_index.hash_tables[table_idx].get(hash_key, []))
            
            # 精确距离计算（限制候选集大小）
            candidates = [c for c in candidates if c[0] == b]  # 同batch内
            if len(candidates) > 0:
                points = x[b, [c[1] for c in candidates]]  # (M, C)
                dists = torch.norm(points - x[b,n], dim=1)
                _, topk_indices = torch.topk(dists, k, largest=False)
                indices[b,n] = torch.tensor([candidates[i][1] for i in topk_indices])
    
    return indices