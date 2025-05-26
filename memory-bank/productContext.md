# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-05-26 13:08:00 - Log of updates made will be appended as footnotes to the end of this file.

## Project Goal

* RDT (Reverse Distillation Training) 时间序列预测验证框架是一个基于Python和PyTorch的时间序列预测验证平台，其核心目标是为RDT方法在时间序列预测领域的应用提供一个可复现、可扩展的实验环境，以便系统性地验证其有效性。

## Key Features

* **RDT 训练实现**: 精确实现基于复合损失和Alpha调度的RDT核心训练逻辑
* **全面的基线对比**: 支持训练并对比RDT模型与标准教师模型、Task-Only学生模型(α=1)、Follower学生模型(α=0)的性能
* **灵活的模型集成**: 方便地引入和配置基于PyTorch的自定义或来自第三方库（如neuralforecast）的其他时间序列模型
* **模块化与可配置性**: 清晰的代码结构(数据、模型、训练器、评估器等)和中心化的配置(src/config.py)
* **标准数据处理流程**: 包含时间序列数据加载、预处理、划分和DataLoader构建
* **Alpha调度策略**: 支持多种Alpha动态调整策略(linear, exponential)或固定值(constant)
* **多维度评估**: 计算常用预测指标、支持鲁棒性测试和稳定性分析，包括模型抗噪音效果、数据去噪效果和学生-教师模型相似度评估
* **丰富的可视化**: 训练/验证损失曲线、预测对比图、性能指标图表等
* **自动化结果管理**: 实验结果自动保存到结构化的结果目录
* **动态Alpha调度增强**: 允许基于验证集性能和模型输出相似度动态调整Alpha权重

## Overall Architecture

* **核心概念**:
  * 教师模型(Teacher Model): 结构简单、训练稳定的模型(如DLinear)
  * 学生模型(Student Model): 容量大、表达能力强的模型(如PatchTST)
  * 复合损失函数: L_total = α·L_task(Y_student, Y_true) + (1-α)·L_distill(Y_student, Y_teacher)
  * Alpha调度: 在训练过程中动态调整α值的策略

* **主要组件**:
  * 数据处理(data_handler.py): 负责数据加载、预处理、划分和DataLoader创建
  * 模型定义(models.py): 定义和实例化各种时间序列预测模型
  * 训练器(trainer.py): 实现标准训练和RDT训练逻辑
  * 评估器(evaluator.py): 负责模型评估、指标计算和鲁棒性测试
  * 调度器(schedulers.py): 实现Alpha调度策略
  * 配置(config.py): 集中管理所有实验参数
  * 工具函数(utils.py): 提供辅助功能如种子设置、绘图、保存加载等