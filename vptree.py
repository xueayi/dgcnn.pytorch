#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VP-Tree implementation for efficient KNN search with Manhattan distance
Optimized for FPGA deployment in DGCNN model
"""

import torch
import numpy as np
import random
from collections import namedtuple
from typing import List, Tuple, Optional, Union

# 定义VP-Tree节点
VPNode = namedtuple('VPNode', ['vantage_point', 'threshold', 'left', 'right'])

class VPTree:
    """
    VP-Tree implementation using Manhattan distance for KNN search
    Optimized for PyTorch tensors and FPGA deployment
    """
    def __init__(self, points: torch.Tensor = None, leaf_size: int = 5):
        """
        Initialize VP-Tree
        
        Args:
            points: Tensor of shape [num_points, dim] containing the points
            leaf_size: Maximum number of points in a leaf node
        """
        self.leaf_size = leaf_size
        self.root = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if points is not None:
            self.build(points)
    
    @staticmethod
    def manhattan_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate Manhattan distance between two points or batches of points
        
        Args:
            x: Tensor of shape [..., dim]
            y: Tensor of shape [..., dim]
            
        Returns:
            Tensor of shape [...] containing the Manhattan distances
        """
        return torch.sum(torch.abs(x - y), dim=-1)
    
    def build(self, points: torch.Tensor):
        """
        Build VP-Tree from points
        
        Args:
            points: Tensor of shape [num_points, dim] containing the points
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=self.device)
        
        # 确保points在正确的设备上
        points = points.to(self.device)
        
        # 构建树
        self.root = self._build_recursive(points)
    
    def _build_recursive(self, points: torch.Tensor):
        """
        Recursively build VP-Tree
        
        Args:
            points: Tensor of shape [num_points, dim] containing the points
            
        Returns:
            VPNode or list of points if leaf node
        """
        n = points.shape[0]
        
        # 如果点数小于等于leaf_size，则创建叶节点
        if n <= self.leaf_size:
            return points
        
        # 随机选择一个vantage point
        vp_idx = random.randrange(n)
        vantage_point = points[vp_idx]
        
        # 计算其他点到vantage point的距离
        points_without_vp = torch.cat([points[:vp_idx], points[vp_idx+1:]])
        distances = self.manhattan_distance(points_without_vp, vantage_point)
        
        # 根据距离中位数分割点集
        median_idx = len(distances) // 2
        threshold = torch.kthvalue(distances, median_idx).values
        
        # 分割点集
        left_points = points_without_vp[distances <= threshold]
        right_points = points_without_vp[distances > threshold]
        
        # 递归构建左右子树
        left = self._build_recursive(left_points)
        right = self._build_recursive(right_points)
        
        return VPNode(vantage_point=vantage_point, threshold=threshold, left=left, right=right)
    
    def _search_knn_recursive(self, node, query_point: torch.Tensor, k: int, results: List[Tuple[float, torch.Tensor]]):
        """
        Recursively search for k nearest neighbors
        
        Args:
            node: Current node (VPNode or points)
            query_point: Query point
            k: Number of nearest neighbors to find
            results: List of (distance, point) tuples, will be modified in-place
        """
        # 如果是叶节点
        if not isinstance(node, VPNode):
            for point in node:
                dist = self.manhattan_distance(query_point, point).item()
                results.append((dist, point))
                results.sort(key=lambda x: x[0])  # 按距离排序
                if len(results) > k:
                    results.pop()  # 移除最远的点
            return
        
        # 计算查询点到vantage point的距离
        dist_to_vp = self.manhattan_distance(query_point, node.vantage_point).item()
        
        # 将vantage point添加到结果中
        results.append((dist_to_vp, node.vantage_point))
        results.sort(key=lambda x: x[0])
        if len(results) > k:
            results.pop()
        
        # 确定先搜索哪个子树
        if dist_to_vp <= node.threshold:
            # 先搜索左子树，再搜索右子树
            if node.left is not None:
                self._search_knn_recursive(node.left, query_point, k, results)
            
            # 如果结果不足k个或者到分界面的距离小于当前第k远的距离，则搜索右子树
            if (len(results) < k or abs(dist_to_vp - node.threshold) < results[-1][0]) and node.right is not None:
                self._search_knn_recursive(node.right, query_point, k, results)
        else:
            # 先搜索右子树，再搜索左子树
            if node.right is not None:
                self._search_knn_recursive(node.right, query_point, k, results)
            
            # 如果结果不足k个或者到分界面的距离小于当前第k远的距离，则搜索左子树
            if (len(results) < k or abs(dist_to_vp - node.threshold) < results[-1][0]) and node.left is not None:
                self._search_knn_recursive(node.left, query_point, k, results)
    
    def query(self, query_points: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query k nearest neighbors for each query point
        
        Args:
            query_points: Tensor of shape [num_queries, dim] containing query points
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (distances, indices) tensors
            distances: Tensor of shape [num_queries, k] containing distances
            indices: Tensor of shape [num_queries, k] containing indices
        """
        if not isinstance(query_points, torch.Tensor):
            query_points = torch.tensor(query_points, device=self.device)
        
        query_points = query_points.to(self.device)
        num_queries = query_points.shape[0]
        
        # 存储结果
        all_distances = []
        all_neighbors = []
        
        # 对每个查询点进行搜索
        for i in range(num_queries):
            results = []  # 存储(distance, point)元组
            self._search_knn_recursive(self.root, query_points[i], k, results)
            
            # 提取距离和点
            distances, neighbors = zip(*results)
            all_distances.append(torch.tensor(distances, device=self.device))
            all_neighbors.append(torch.stack(neighbors))
        
        # 将结果转换为张量
        distances_tensor = torch.stack(all_distances)
        neighbors_tensor = torch.stack(all_neighbors)
        
        return distances_tensor, neighbors_tensor

# 批处理版本的VP-Tree KNN搜索，用于DGCNN模型
def vp_tree_knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    VP-Tree based KNN search for DGCNN model
    Uses Manhattan distance instead of Euclidean distance
    
    Args:
        x: Tensor of shape [batch_size, feature_dim, num_points]
        k: Number of nearest neighbors to find
        
    Returns:
        Tensor of shape [batch_size, num_points, k] containing indices of k-nearest neighbors
    """
    batch_size, feature_dim, num_points = x.size()
    device = x.device
    
    # 转置为 [batch_size, num_points, feature_dim]
    x_transposed = x.transpose(2, 1).contiguous()
    
    # 存储所有批次的KNN索引
    all_indices = []
    
    # 对每个批次单独处理
    for batch_idx in range(batch_size):
        # 获取当前批次的点
        points = x_transposed[batch_idx]  # [num_points, feature_dim]
        
        # 构建VP-Tree
        vp_tree = VPTree(points)
        
        # 查询每个点的k个最近邻
        batch_indices = []
        for i in range(num_points):
            query_point = points[i:i+1]  # 保持维度 [1, feature_dim]
            
            # 排除查询点自身
            mask = torch.ones(num_points, dtype=torch.bool, device=device)
            mask[i] = False
            other_points = points[mask]
            
            # 构建不包含查询点的VP-Tree
            query_vp_tree = VPTree(other_points)
            
            # 查询k个最近邻
            _, neighbors = query_vp_tree.query(query_point, k)
            
            # 将邻居点映射回原始索引
            original_indices = torch.where(mask)[0]
            neighbor_indices = original_indices[neighbors.squeeze(0)]
            
            batch_indices.append(neighbor_indices)
        
        # 将当前批次的索引堆叠起来
        batch_indices_tensor = torch.stack(batch_indices)
        all_indices.append(batch_indices_tensor)
    
    # 将所有批次的索引堆叠起来
    indices = torch.stack(all_indices)
    
    return indices

# 优化版本，使用预计算的距离矩阵
def vp_tree_knn_optimized(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Optimized VP-Tree based KNN search for DGCNN model
    Uses Manhattan distance with precomputed distance matrix
    
    Args:
        x: Tensor of shape [batch_size, feature_dim, num_points]
        k: Number of nearest neighbors to find
        
    Returns:
        Tensor of shape [batch_size, num_points, k] containing indices of k-nearest neighbors
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    device = x.device
    
    # 转置为 [batch_size, num_points, feature_dim]
    x = x.transpose(2, 1).contiguous()
    
    # 存储所有批次的KNN索引
    all_indices = []
    
    # 对每个批次单独处理
    for batch_idx in range(batch_size):
        # 获取当前批次的点
        points = x[batch_idx]  # [num_points, feature_dim]
        
        # 计算曼哈顿距离矩阵
        # 使用广播计算所有点对之间的曼哈顿距离
        # 形状: [num_points, num_points]
        dist_matrix = torch.sum(torch.abs(points.unsqueeze(1) - points.unsqueeze(0)), dim=2)
        
        # 将自身距离设为无穷大，以便不选择自身作为最近邻
        dist_matrix.fill_diagonal_(float('inf'))
        
        # 获取每个点的k个最近邻的索引
        # topk返回(values, indices)，我们只需要indices
        _, indices = dist_matrix.topk(k=k, dim=1, largest=False)
        
        all_indices.append(indices)
    
    # 将所有批次的索引堆叠起来
    indices = torch.stack(all_indices)
    
    return indices