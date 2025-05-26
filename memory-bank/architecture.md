# RDT 架构文档

本文档使用图表和说明来描述RDT时间序列预测验证框架的架构设计和组件关系。

## 1. 系统组件关系图

```mermaid
graph TD
    A[main.py] --> B[src/config.py]
    A --> C[src/data_handler.py]
    A --> D[src/models.py]
    A --> E[src/trainer.py]
    A --> F[src/evaluator.py]
    A --> G[src/utils.py]
    A --> H[src/schedulers.py]
    
    B -->|配置参数| C
    B -->|配置参数| D
    B -->|配置参数| E
    B -->|配置参数| F
    B -->|配置参数| H
    
    C -->|DataLoader| E
    C -->|DataLoader| F
    
    D -->|Teacher Model| E
    D -->|Student Model| E
    D -->|Models| F
    
    E -->|Trained Models| F
    H -->|Alpha Scheduler| E
    
    F -->|评估结果| G
    G -->|图表生成| A
    G -->|模型保存/加载| E
```

## 2. RDT训练数据流图

```mermaid
flowchart LR
    IN[输入数据] --> TCH[教师模型]
    IN --> STU[学生模型]
    
    TCH -->|Y_teacher| DIST[蒸馏损失]
    STU -->|Y_student| DIST
    STU -->|Y_student| TASK[任务损失]
    GT[真实标签] -->|Y_true| TASK
    
    DIST -->|L_distill| COMB{复合损失}
    TASK -->|L_task| COMB
    SCH[Alpha调度器] -->|α值| COMB
    
    COMB -->|L_total| OPT[优化器]
    OPT --> STU
```

## 3. 核心类图

```mermaid
classDiagram
    class BaseTrainer {
        +model
        +train_loader
        +val_loader
        +optimizer
        +loss_fn
        +device
        +epochs
        +model_save_path
        +history
        +train()
        #_train_epoch()
        #_validate_epoch()
    }
    
    class StandardTrainer {
        +_train_epoch()
        +_validate_epoch()
    }
    
    class RDT_Trainer {
        +teacher_model
        +distill_loss_fn
        +alpha_scheduler
        +_train_epoch()
        +_validate_epoch()
    }
    
    class BaseAlphaScheduler {
        +get_alpha(epoch)
    }
    
    class LinearScheduler {
        +start_alpha
        +end_alpha
        +epochs
        +get_alpha(epoch)
    }
    
    class ExponentialScheduler {
        +start_alpha
        +end_alpha
        +epochs
        +get_alpha(epoch)
    }
    
    class ConstantScheduler {
        +alpha
        +get_alpha(epoch)
    }
    
    class TimeSeriesDataset {
        +data
        +lookback
        +horizon
        +n_features
        +indices
        +__len__()
        +__getitem__(idx)
    }
    
    BaseTrainer <|-- StandardTrainer
    BaseTrainer <|-- RDT_Trainer
    BaseAlphaScheduler <|-- LinearScheduler
    BaseAlphaScheduler <|-- ExponentialScheduler
    BaseAlphaScheduler <|-- ConstantScheduler
    RDT_Trainer o-- BaseAlphaScheduler
```

## 4. 数据处理流程

```mermaid
flowchart TD
    A[加载原始数据] --> B[日期处理]
    B --> C[目标列提取]
    C --> D[处理缺失值]
    D --> E[数据划分]
    E -->|训练数据| F[数据标准化拟合]
    F -->|训练数据| G[创建训练Dataset]
    F -->|验证数据| H[创建验证Dataset]
    F -->|测试数据| I[创建测试Dataset]
    G --> J[创建训练DataLoader]
    H --> K[创建验证DataLoader]
    I --> L[创建测试DataLoader]
```

## 5. 模型实例化流程

```mermaid
flowchart TD
    A[配置模型参数] --> B{选择模型类型}
    B -->|DLinear| C[实例化DLinear]
    B -->|PatchTST| D[实例化PatchTST]
    B -->|NLinear| E[实例化NLinear]
    B -->|MLP| F[实例化MLPModel]
    B -->|RNN| G[实例化RNNModel]
    B -->|LSTM| H[实例化LSTMModel]
    B -->|Autoformer| I[实例化Autoformer]
    B -->|Informer| J[实例化Informer]
    B -->|FEDformer| K[实例化FEDformer]
    
    C --> L[返回模型实例]
    D --> L
    E --> L
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
```

## 6. 评估流程

```mermaid
flowchart TD
    A[加载模型] --> B[在测试集上评估]
    B --> C[计算MSE/MAE等指标]
    B --> D[生成预测vs真实值图表]
    C --> E[输出性能报告]
    D --> E
    
    B --> F[添加噪声进行鲁棒性测试]
    F --> G[不同噪声级别下计算指标]
    G --> H[生成鲁棒性曲线]
    H --> E
    
    A --> I[多次运行]
    I --> J[收集多次运行指标]
    J --> K[生成稳定性分析图表]
    K --> E