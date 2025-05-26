# Decision Log

This file records architectural and implementation decisions using a list format.
2025-05-26 13:10:00 - Log of updates made.

## Decision

* **核心架构基于RDT理念设计**

## Rationale 

* 传统知识蒸馏使用复杂模型（教师）指导简单模型（学生），而RDT提出反向思路：利用结构简单、训练稳定的时间序列模型作为教师，去指导结构复杂、容量更大的现代深度学习模型作为学生。
* 这种方法可以提升稳定性与鲁棒性，隐式注入先验知识，并通过动态平衡拟合真实数据和模仿教师模型之间的关系。

## Implementation Details

* 实现了复合损失函数：L_total = α·L_task + (1-α)·L_distill，其中α可动态调整
* 使用DLinear作为默认教师模型，PatchTST作为默认学生模型
* 支持多种Alpha调度策略：线性、指数和常数
* 设计了模块化架构，包括数据处理、模型定义、训练器和评估器

---

## Decision

* **标准化数据处理流程**

## Rationale 

* 时间序列数据预处理对模型性能至关重要
* 不同数据集可能有不同的特性，需要统一的处理框架

## Implementation Details

* 实现了TimeSeriesDataset类和专用DataLoader
* 支持时间索引、缺失值处理、数据划分和标准化
* 使用StandardScaler对训练集拟合，并应用于验证集和测试集
* 设计了滑动窗口方法来生成特定长度的输入序列和预测目标

---

## Decision

* **支持多种时间序列模型**

## Rationale 

* 不同模型在不同数据集上可能表现各异
* 需要灵活的框架来比较不同模型组合的性能

## Implementation Details

* 集成了neuralforecast库中的多种模型，如DLinear、NLinear、PatchTST等
* 自定义实现了MLP、RNN和LSTM模型
* 设计了统一的模型接口，支持一致的训练和评估流程
* 提供了模型工厂函数（get_model）以简化模型实例化