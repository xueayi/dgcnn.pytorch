# DGCNN与Point-NN融合使用说明

本文档说明如何使用DGCNN与Point-NN的直接预测融合功能。

## 背景

根据论文内容，Point-NN的Plug-and-Play方法可以通过直接融合预测结果的方式无缝融入DGCNN，无需修改DGCNN的原有结构或重新训练。

## 实现原理

### 直接预测融合（Inference-Time Ensemble）

1. **独立运行两个模型**：
   - 对同一输入点云，分别用DGCNN和Point-NN进行前向推理，得到各自的分类logits。
   - DGCNN输出：logits_DGCNN ∈ ℝᴷ（K为类别数）。
   - Point-NN输出：logits_Point-NN ∈ ℝᴷ（通过点记忆库相似度匹配生成）。

2. **线性插值融合**：
   - 通过加权平均融合两者的输出：
     ```
     logits_Final = λ · logits_DGCNN + (1-λ) · logits_Point-NN
     ```
     其中λ ∈ [0,1]为权重超参数（默认λ=0.5）。

3. **后处理**：
   - 对logits_Final应用Softmax得到最终预测类别。

## 使用方法

### 前提条件

1. 确保已安装DGCNN和Point-NN的所有依赖项
2. 确保Point-NN项目位于与DGCNN同级的目录中

### 运行命令

```bash
python main_cls.py --exp_name=cls_0327_eca_nn_eval --num_points=1024 --k=20 --eval=True --use_point_nn=True --lambda_weight=0.5 --model_path=outputs/exp_0327_eca/models/model.t7
```

### 参数说明

- `--eval`: 进行评估而非训练
- `--model`: 选择基础模型，可选值为'pointnet'或'dgcnn'
- `--model_path`: 预训练模型的路径
- `--use_point_nn`: 启用Point-NN融合
- `--lambda_weight`: 融合权重λ，控制DGCNN的权重（默认0.5）

## 优势

- **零训练成本**：无需调整DGCNN的参数。
- **几何-语义互补**：  
  - DGCNN擅长捕捉高阶语义特征（通过动态图卷积）。
  - Point-NN专注于低频几何结构（如尖锐边缘、局部形状变化）。

## 调参建议

- 在验证集上调整λ（如λ=0.7偏向DGCNN，λ=0.3偏向Point-NN）。
- 由于Point-NN无需训练，可预计算其特征缓存，减少实时推理开销。