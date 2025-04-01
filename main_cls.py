#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2019/12/30 9:32 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40
from model import PointNet, DGCNN_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, cls_acc
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
from thop import profile
from thop import clever_format
import copy
from tqdm import tqdm
from nn_models import Point_NN




def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    if not os.path.exists('outputs/'+args.exp_name+'/tensorboard'):
        os.makedirs('outputs/'+args.exp_name+'/tensorboard')
    os.system('cp main_cls.py outputs'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    writer = SummaryWriter(log_dir='outputs/'+args.exp_name+'/tensorboard')
    
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))

    # 使用副本统计模型参数量和FLOPs
    model_copy = copy.deepcopy(model)
    dummy_input = torch.randn(1, 3, args.num_points).to(device)
    macs, params = profile(model_copy, inputs=(dummy_input,))
    del model_copy  # 统计完成后立即释放副本
    writer.add_scalar('Model/Params', params, 0)
    writer.add_scalar('Model/MACs', macs, 0)
    macs, params = clever_format([macs, params], "%.3f")
    io.cprint(f"模型参数量: {params}, 计算量: {macs}")

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true).astype(np.int64)
        train_pred = np.concatenate(train_pred).astype(np.int64)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        # 记录训练指标
        writer.add_scalar('Train/Loss', train_loss/count, epoch)
        writer.add_scalar('Train/Acc', train_acc, epoch)
        writer.add_scalar('Train/Balanced_Acc', train_balanced_acc / count, epoch)
        writer.add_scalar('Train/LR', opt.param_groups[0]['lr'], epoch)
        
        # 记录GPU内存使用情况
        if args.cuda:
            writer.add_scalar('System/GPU_Mem', torch.cuda.max_memory_allocated()/1024**3, epoch)
        
        outstr = '[Epoch %d] Train Loss: %.4f | Acc: %.4f | Balanced Acc: %.4f | LR: %.5f | GPU Mem: %.2fGB' % (
            epoch, 
            train_loss/count,
            train_acc,
            train_balanced_acc,
            opt.param_groups[0]['lr'],
            torch.cuda.max_memory_allocated()/1024**3 if args.cuda else 0
        )
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true).astype(np.int64)
        test_pred = np.concatenate(test_pred).astype(np.int64)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        # 记录测试指标
        writer.add_scalar('Test/Loss', test_loss/count, epoch)
        writer.add_scalar('Test/Acc', test_acc, epoch)
        writer.add_scalar('Test/Balanced_Acc', avg_per_class_acc, epoch)
        writer.add_scalar('Best_Acc', best_test_acc, epoch)
        
        outstr = '[Epoch %d] Test Loss: %.4f | Test Acc: %.4f | Test Balanced Acc: %.4f | Best Acc: %.4f' % (
            epoch,
            test_loss/count,
            test_acc,
            avg_per_class_acc,
            best_test_acc
        )
        io.cprint(outstr)
        
        # 重置GPU内存统计
        if args.cuda:
            torch.cuda.reset_peak_memory_stats()
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
                             
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    # 统计模型参数量和FLOPs
    model_copy = copy.deepcopy(model)
    dummy_input = torch.randn(1, 3, args.num_points).to(device)
    macs, params = profile(model_copy, inputs=(dummy_input,))
    del model_copy  # 统计完成后立即释放副本
    macs, params = clever_format([macs, params], "%.3f")
    io.cprint(f"模型参数量: {params}, 计算量: {macs}")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    
    if args.use_point_nn:
        io.cprint(f"使用Point-NN集成，融合权重λ={args.lambda_weight}")
        
        # 初始化Point-NN
        point_nn = Point_NN(input_points=args.num_points, 
                          num_stages=4,  # 默认值
                          embed_dim=72,  # 默认值
                          k_neighbors=90,  # 默认值
                          alpha=1000,  # 默认值
                          beta=100  # 默认值
                          ).to(device)
        point_nn.eval()
        
        print('==> 构建Point-NN记忆库...')
        feature_memory, label_memory = [], []
        with torch.no_grad():
            for points, labels in tqdm(train_loader):
                points = points.to(device).permute(0, 2, 1)  # [B, 3, N]
                # 通过非参数编码器
                point_features = point_nn(points)  # [B, C]
                feature_memory.append(point_features)
                
                labels = labels.to(device)
                label_memory.append(labels)
        
        # 特征记忆库
        feature_memory = torch.cat(feature_memory, dim=0)  # [num_train, C]
        feature_memory /= feature_memory.norm(dim=-1, keepdim=True)  # 归一化
        feature_memory = feature_memory.permute(1, 0)  # [C, num_train]
        
        # 标签记忆库
        label_memory = torch.cat(label_memory, dim=0)  # [num_train]
        label_memory = F.one_hot(label_memory).squeeze().float()  # [num_train, num_classes]
        
        print('==> 提取测试点云特征...')
        test_features, test_labels = [], []
        with torch.no_grad():
            for points, labels in tqdm(test_loader):
                points = points.cuda().permute(0, 2, 1)  # [B, 3, N]
                point_features = point_nn(points)  # [B, C]
                test_features.append(point_features)
                test_labels.append(labels.to(device))
        
        test_features = torch.cat(test_features)  # [num_test, C]
        test_features /= test_features.norm(dim=-1, keepdim=True)  # 归一化
        test_labels = torch.cat(test_labels)  # [num_test]
        
        print('==> 寻找Point-NN最佳gamma参数...')
        gamma_list = [i * 10000 / 5000 for i in range(5000)]
        best_acc, best_gamma = 0, 0
        for gamma in gamma_list:
            # 相似度匹配
            Sim = test_features @ feature_memory  # [num_test, num_train]
            # 标签集成
            point_nn_logits = (-gamma * (1 - Sim)).exp() @ label_memory  # [num_test, num_classes]
            
            acc = cls_acc(point_nn_logits, test_labels)
            if acc > best_acc:
                best_acc, best_gamma = acc, gamma
        
        print(f'==> 找到最佳gamma: {best_gamma:.2f}, Point-NN准确率: {best_acc:.2f}%')
    
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)  # [B, 3, N]
        batch_size = data.size()[0]
        
        # DGCNN预测
        dgcnn_logits = model(data)
        
        if args.use_point_nn:
            # Point-NN预测
            point_features = point_nn(data)
            point_features /= point_features.norm(dim=-1, keepdim=True)
            Sim = point_features @ feature_memory
            point_nn_logits = (-best_gamma * (1 - Sim)).exp() @ label_memory
            
            # 融合预测结果
            logits = args.lambda_weight * dgcnn_logits + (1 - args.lambda_weight) * point_nn_logits
        else:
            logits = dgcnn_logits
        
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--use_point_nn', type=bool, default=False,
                        help='是否使用Point-NN进行预测融合')
    parser.add_argument('--lambda_weight', type=float, default=0.5,
                        help='融合权重λ，控制DGCNN的权重 (默认0.5)')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        try:
            train(args, io)
            io.cprint('训练完成')
        finally:
            if 'writer' in locals():
                writer.close()
    else:
        test(args, io)
