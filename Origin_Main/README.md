# Reverse Distillation Transformer (RDT)

这个项目实现了一个反向蒸馏变换器（Reverse Distillation Transformer）模型，用于时间序列预测任务。该项目从原始的Jupyter notebook重构为一个模块化的Python项目，提高了代码的可读性和可维护性。

## 项目结构

项目分为以下几个主要模块：

- `config.py`: 包含所有配置参数，如数据路径、模型超参数等
- `data_preprocessing.py`: 负责数据加载、清洗和预处理
- `models.py`: 定义了线性教师模型和Transformer学生模型
- `training.py`: 实现了反向蒸馏训练逻辑
- `inference.py`: 处理模型推理和预测
- `evaluation.py`: 提供评估指标计算和可视化功能
- `main.py`: 主程序，整合所有模块并执行完整的训练和评估流程

## 使用方法

1. 确保已安装所有必要的依赖项：
   ```
   pip install torch numpy pandas matplotlib scikit-learn
   ```

2. 确保数据文件 `weatherHistory.csv` 位于项目根目录

3. 运行主程序：
   ```
   python main.py
   ```

## 模型说明

该项目实现了反向蒸馏技术，使用线性回归模型作为教师模型，Transformer作为学生模型。训练过程中，学生模型同时学习真实标签和教师模型的软标签，通过动态调整两种损失的权重来平衡学习过程。

主要特点：
- 使用过去24小时的数据预测未来3小时的温度
- 线性教师模型提供软标签指导学习
- Transformer学生模型捕获时间序列的复杂模式
- 动态调整任务损失和蒸馏损失的权重

## 结果可视化

训练完成后，程序会生成预测结果的可视化图表，并保存为PNG文件：
- `rdt_transformer_prediction_vs_actual.png`: RDT模型预测结果
- `linear_teacher_prediction_vs_actual.png`: 线性教师模型预测结果
