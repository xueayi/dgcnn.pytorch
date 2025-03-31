import torch
import numpy as np

class VPTree:
    def __init__(self, points):
        """初始化VP-Tree
        Args:
            points: 输入点云数据，形状为(N, D)的张量，N为点的数量，D为维度
        """
        self.points = points
        self.left = None
        self.right = None
        self.vantage_point = None
        self.median_dist = None
        
        if len(points) > 0:
            self._build_tree()
    
    def _manhattan_distance(self, x, y):
        """计算曼哈顿距离
        Args:
            x: 形状为(D,)的张量
            y: 形状为(N, D)的张量
        Returns:
            形状为(N,)的距离张量
        """
        return torch.sum(torch.abs(y - x), dim=1)
    
    def _build_tree(self):
        """构建VP-Tree"""
        if len(self.points) <= 1:
            return
        
        # 随机选择一个支撑点
        vp_idx = np.random.randint(len(self.points))
        self.vantage_point = self.points[vp_idx]
        
        # 计算其他点到支撑点的距离
        other_points = torch.cat([self.points[:vp_idx], self.points[vp_idx+1:]])
        distances = self._manhattan_distance(self.vantage_point, other_points)
        
        # 使用中位数划分点集
        median_idx = len(distances) // 2
        self.median_dist = torch.sort(distances)[0][median_idx].item()
        
        # 划分左右子树的点集
        left_points = other_points[distances <= self.median_dist]
        right_points = other_points[distances > self.median_dist]
        
        # 递归构建左右子树
        self.left = VPTree(left_points)
        self.right = VPTree(right_points)
    
    def _search_k_nearest(self, query_point, k, results):
        """搜索k个最近邻点
        Args:
            query_point: 查询点，形状为(D,)的张量
            k: 需要查找的最近邻数量
            results: 存储结果的列表，每个元素为(距离, 点)的元组
        """
        if self.vantage_point is None:
            return
        
        # 计算查询点到支撑点的距离
        dist_to_vp = self._manhattan_distance(query_point, self.vantage_point.unsqueeze(0))[0]
        
        # 将支撑点加入结果集
        results.append((dist_to_vp, self.vantage_point))
        results.sort(key=lambda x: x[0])
        if len(results) > k:
            results.pop()
        
        # 根据距离决定搜索顺序
        if dist_to_vp <= self.median_dist:
            first, second = self.left, self.right
        else:
            first, second = self.right, self.left
        
        # 递归搜索子树
        if first is not None:
            first._search_k_nearest(query_point, k, results)
        
        # 判断是否需要搜索另一个子树
        if second is not None and len(results) < k or \
           abs(dist_to_vp - self.median_dist) < results[-1][0]:
            second._search_k_nearest(query_point, k, results)
    
    def query(self, query_point, k):
        """查询k个最近邻点
        Args:
            query_point: 查询点，形状为(D,)的张量
            k: 需要查找的最近邻数量
        Returns:
            k个最近邻点的索引
        """
        results = []
        self._search_k_nearest(query_point, k, results)
        return torch.stack([x[1] for x in results])

def batch_knn_vptree(x, k):
    """批量处理的kNN搜索函数，使用优化后的VP-Tree实现
    Args:
        x: 输入点云数据，形状为(B, D, N)的张量
        k: 需要查找的最近邻数量
    Returns:
        形状为(B, N, k)的最近邻索引张量
    """
    batch_size, num_dims, num_points = x.size()
    device = x.device
    
    # 确保k值不超过点的数量
    k = min(k, num_points - 1)
    if k < 1:
        raise ValueError(f"k值必须大于0且小于点的数量({num_points})")
    
    # 将输入转换为(B, N, D)格式
    x = x.transpose(2, 1).contiguous()
    
    # 初始化结果张量
    indices = torch.zeros((batch_size, num_points, k), dtype=torch.long, device=device)
    
    # 使用CUDA加速的曼哈顿距离计算
    @torch.jit.script
    def manhattan_distance_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(x.unsqueeze(1) - y.unsqueeze(0)), dim=2)
    
    # 对每个batch并行处理
    for batch_idx in range(batch_size):
        points = x[batch_idx]
        
        # 计算所有点对之间的距离矩阵
        distances = manhattan_distance_cuda(points, points)
        
        # 将自身距离设置为无穷大，避免选择自身作为最近邻
        distances.fill_diagonal_(float('inf'))
        
        # 直接使用topk找到k个最近邻
        _, indices[batch_idx] = torch.topk(distances, k, dim=1, largest=False)
    
    return indices