# System Patterns

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.
2025-05-26 13:10:00 - Log of updates made.

## Coding Patterns

* **工厂函数模式**
  * `get_model()` 函数用于实例化各种模型类型
  * `get_optimizer()` 和 `get_loss_function()` 用于创建优化器和损失函数
  * 通过配置参数控制具体实例化的对象

* **继承模式**
  * `BaseTrainer` 作为基类，`StandardTrainer` 和 `RDT_Trainer` 继承并实现特定逻辑
  * 抽象方法 `_train_epoch` 和 `_validate_epoch` 由子类实现

* **组合模式**
  * 训练器组合了模型、数据加载器、优化器和损失函数
  * 整体系统组合了数据处理、模型、训练器和评估器组件

* **回调模式**
  * `EarlyStopping` 类实现了训练过程中的回调机制
  * 通过监控验证损失来决定是否提前终止训练

## Architectural Patterns

* **模块化设计**
  * 核心功能被划分为明确的模块：models.py, trainer.py, data_handler.py, evaluator.py
  * 配置集中管理在config.py中，便于全局调整

* **依赖注入**
  * 通过参数将依赖项(如模型、数据加载器)传递给训练器和评估器
  * 避免了组件间的紧耦合，提高了可测试性和灵活性

* **策略模式**
  * 不同的训练方法(标准、RDT)作为不同的策略实现
  * Alpha调度器采用策略模式，支持多种调度策略(线性、指数、常数)

* **数据流模式**
  * 清晰的数据处理流：加载 -> 预处理 -> 划分 -> 标准化 -> DataLoader
  * 训练数据流：输入 -> 模型前向传播 -> 损失计算 -> 反向传播 -> 参数更新

## Testing Patterns

* **基线比较测试**
  * 通过比较不同模型配置的性能来评估新方法
  * 包括教师模型、Task-Only学生模型(α=1)和Follower学生模型(α=0)作为基线

* **鲁棒性测试**
  * 通过向测试数据添加不同级别的噪声来评估模型的鲁棒性
  * 使用多个指标(MSE, MAE)全面评估模型性能

* **多次运行测试**
  * 支持多次运行实验以分析结果的稳定性
  * 使用箱线图/小提琴图可视化多次运行的性能分布