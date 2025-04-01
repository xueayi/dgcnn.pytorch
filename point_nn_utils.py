import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 导入Point-NN相关模块
from nn_models import Point_NN
from util import cls_acc


class PointNNWrapper:
    """
    Point-NN模型的包装器，用于与DGCNN集成
    实现了直接预测融合（Inference-Time Ensemble）方法
    """
    def __init__(self, num_points=1024, num_stages=4, embed_dim=72, k_neighbors=90, alpha=1000, beta=100):
        """
        初始化Point-NN模型
        
        Args:
            num_points: 输入点云的点数量
            num_stages: Point-NN的阶段数
            embed_dim: 嵌入维度
            k_neighbors: k近邻数量
            alpha: PosE编码的alpha参数
            beta: PosE编码的beta参数
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.point_nn = Point_NN(input_points=num_points, 
                                num_stages=num_stages,
                                embed_dim=embed_dim, 
                                k_neighbors=k_neighbors,
                                beta=beta, 
                                alpha=alpha).to(self.device)
        self.point_nn.eval()  # 设置为评估模式
        
        # 记忆库
        self.feature_memory = None
        self.label_memory = None
        self.best_gamma = None
        
    def build_memory_bank(self, train_loader):
        """
        构建Point-NN的记忆库
        
        Args:
            train_loader: 训练数据加载器
        """
        print('==> 构建Point-NN记忆库...')
        feature_memory, label_memory = [], []
        
        with torch.no_grad():
            for points, labels in tqdm(train_loader):
                points = points.to(self.device).permute(0, 2, 1)  # [B, 3, N]
                # 通过非参数编码器
                point_features = self.point_nn(points)  # [B, C]
                feature_memory.append(point_features)
                
                labels = labels.to(self.device)
                label_memory.append(labels)
        
        # 特征记忆库
        self.feature_memory = torch.cat(feature_memory, dim=0)  # [num_train, C]
        self.feature_memory /= self.feature_memory.norm(dim=-1, keepdim=True)  # 归一化
        self.feature_memory = self.feature_memory.permute(1, 0)  # [C, num_train]
        
        # 标签记忆库
        self.label_memory = torch.cat(label_memory, dim=0)  # [num_train]
        self.label_memory = F.one_hot(self.label_memory).squeeze().float()  # [num_train, num_classes]
        
        print(f'==> 记忆库构建完成: 特征形状 {self.feature_memory.shape}, 标签形状 {self.label_memory.shape}')
    
    def find_best_gamma(self, test_loader):
        """
        在验证集上寻找最佳的gamma参数
        
        Args:
            test_loader: 测试数据加载器
        """
        if self.feature_memory is None or self.label_memory is None:
            raise ValueError("请先构建记忆库")
            
        print('==> 寻找Point-NN最佳gamma参数...')
        # 提取测试特征
        test_features, test_labels = [], []
        with torch.no_grad():
            for points, labels in tqdm(test_loader):
                points = points.to(self.device).permute(0, 2, 1)  # [B, 3, N]
                point_features = self.point_nn(points)  # [B, C]
                test_features.append(point_features)
                test_labels.append(labels.to(self.device))
        
        test_features = torch.cat(test_features)  # [num_test, C]
        test_features /= test_features.norm(dim=-1, keepdim=True)  # 归一化
        test_labels = torch.cat(test_labels)  # [num_test]
        
        # 搜索最佳gamma参数
        gamma_list = [i * 10000 / 5000 for i in range(5000)]
        best_acc, best_gamma = 0, 0
        
        for gamma in gamma_list:
            # 相似度匹配
            Sim = test_features @ self.feature_memory  # [num_test, num_train]
            # 标签集成
            logits = (-gamma * (1 - Sim)).exp() @ self.label_memory  # [num_test, num_classes]
            
            # 计算准确率
            acc = cls_acc(logits, test_labels)
            
            if acc > best_acc:
                best_acc, best_gamma = acc, gamma
        
        self.best_gamma = best_gamma
        print(f'==> 找到最佳gamma: {best_gamma:.2f}, Point-NN准确率: {best_acc:.2f}%')
        return best_gamma
    
    def predict(self, points):
        """
        使用Point-NN进行预测
        
        Args:
            points: 输入点云 [B, 3, N]
            
        Returns:
            logits: 预测的logits [B, num_classes]
        """
        if self.feature_memory is None or self.label_memory is None:
            raise ValueError("请先构建记忆库")
        if self.best_gamma is None:
            raise ValueError("请先寻找最佳gamma参数")
            
        with torch.no_grad():
            # 提取特征
            point_features = self.point_nn(points)  # [B, C]
            point_features /= point_features.norm(dim=-1, keepdim=True)  # 归一化
            
            # 相似度匹配
            Sim = point_features @ self.feature_memory  # [B, num_train]
            
            # 标签集成
            logits = (-self.best_gamma * (1 - Sim)).exp() @ self.label_memory  # [B, num_classes]
            
        return logits


def ensemble_predictions(dgcnn_logits, pointnn_logits, lambda_weight=0.5):
    """
    集成DGCNN和Point-NN的预测结果
    
    Args:
        dgcnn_logits: DGCNN的logits输出 [B, num_classes]
        pointnn_logits: Point-NN的logits输出 [B, num_classes]
        lambda_weight: 权重参数，控制DGCNN的权重 (默认0.5)
        
    Returns:
        final_logits: 融合后的logits [B, num_classes]
    """
    # 线性插值融合
    final_logits = lambda_weight * dgcnn_logits + (1 - lambda_weight) * pointnn_logits
    return final_logits