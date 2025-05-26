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
---

## Decision

*   **在 `src/config.py` 中增加了噪音注入、数据平滑和模型相似度评估的配置项**

## Rationale

*   为了支持新功能，包括模型抗噪音效果评估、数据去噪效果评估和学生-教师模型相似度评估，以及动态 Alpha 调度。

## Implementation Details

*   `NOISE_INJECTION_LEVELS` 和 `NOISE_TYPE` 用于数据增强和鲁棒性测试。
*   `SMOOTHING_METHOD` 和 `SMOOTHING_FACTOR` 用于数据预处理。
*   `SIMILARITY_METRIC` 用于学生-教师模型输出的相似度计算。
---

## Decision

*   **在 `src/data_handler.py` 中增加了噪音注入和数据平滑/合成的逻辑**

## Rationale

*   为了支持数据增强和鲁棒性测试，以及提供数据预处理的灵活性。

## Implementation Details

*   添加了 `add_noise` 函数，支持高斯、椒盐和泊松噪声。
*   添加了 `smooth_data` 函数，支持移动平均和指数平滑。
*   修改了 `load_and_preprocess_data` 函数，在数据标准化之前应用平滑，在标准化之后对训练数据应用噪音注入。
---

## Decision

*   **在 `src/evaluator.py` 中增加了余弦相似度计算函数和 `calculate_similarity_metrics` 函数，并修改 `evaluate_model` 以集成学生-教师相似度评估**

## Rationale

*   为了支持学生模型和教师模型预测结果之间的相似度评估，这是动态 Alpha 调度的一个重要输入。

## Implementation Details

*   添加了 `cosine_similarity` 函数用于计算余弦相似度。
*   添加了 `calculate_similarity_metrics` 函数，根据配置的相似度指标计算学生和教师预测之间的相似度。
*   修改了 `evaluate_model` 函数，使其接受可选的 `teacher_predictions_original` 参数，并在提供时调用 `calculate_similarity_metrics`。
---

## Decision

*   **修改 `src/trainer.py` 中的 `RDT_Trainer`，使其在验证阶段收集学生模型、教师模型和真实标签的预测结果，并将其传递给 Alpha 调度器进行动态调整**

## Rationale

*   为了实现基于验证集信息（包括模型相似度）动态调整 Alpha 权重的功能，Alpha 调度器需要访问这些预测结果。

## Implementation Details

*   在 `_validate_epoch` 方法中，除了计算任务损失外，还收集了学生模型预测 (`batch_y_student`)、教师模型预测 (`batch_y_teacher`) 和真实标签 (`batch_y_true`)。
*   这些收集到的预测结果在每个验证 epoch 结束时被传递给 `self.alpha_scheduler.update()` 方法。
---

## Decision

*   **修改 `src/schedulers.py` 中的 `AlphaScheduler` 基类及其子类，增加 `update` 方法，允许基于验证集信息（包括模型相似度）动态调整 Alpha 权重**

## Rationale

*   为了实现更智能的 Alpha 调度策略，允许调度器根据训练过程中的实时反馈（如验证损失和模型预测相似度）来调整 Alpha 值。

## Implementation Details

*   在 `BaseAlphaScheduler` 中添加了 `update` 抽象方法，接收 `current_epoch`, `val_loss`, `student_preds`, `teacher_preds`, `true_labels` 作为参数。
*   在 `ConstantScheduler`, `LinearScheduler`, `ExponentialScheduler` 的 `update` 方法中，目前只是一个 `pass` 操作，表示这些调度器不进行动态调整。未来可以根据需求实现更复杂的动态调度策略。

---

## Decision

*   **在 `src/evaluator.py` 的 `predict` 函数中添加类型检查**

## Rationale

*   为了增强 `predict` 函数的健壮性，确保其接收到的 `model` 参数确实是一个 PyTorch 模型实例。
*   在 `run_quick_test_evaluation.py` 运行时，`predict` 函数在调用 `model.eval()` 时，`model` 参数有时会是一个 `numpy.ndarray` 对象，导致 `AttributeError`。这表明调用方错误地传递了参数。通过添加类型检查，可以在更早的阶段捕获此类错误，并提供更清晰的错误信息。

## Implementation Details

*   在 `predict` 函数的开头添加了 `if not isinstance(model, torch.nn.Module):` 检查。
*   如果类型不匹配，则抛出 `TypeError`，并提供详细的错误消息。

---

## Decision

*   **优化实验结果文件夹命名，包含随机种子和运行索引**

## Rationale

*   为了提高实验结果的复现性，确保每次稳定性运行（Stability Run）的结果都能追溯到其使用的随机种子。
*   避免在 `STABILITY_RUNS` 存在时，多个运行结果存储在同一个父文件夹下，导致文件重复或覆盖。

## Implementation Details

*   在 `run_evaluation_experiments.py` 的 `run_experiment` 函数中，为每个稳定性运行创建了一个新的子文件夹。
*   子文件夹的命名格式为 `runX_seed_Y`，其中 `X` 是运行的索引（从0开始），`Y` 是该运行使用的随机种子。
*   所有与该次运行相关的模型、绘图和日志文件都将保存到这个新的子文件夹中。

---

## Decision

*   **优化和规范化实验结果保存结构，采用多层级目录结构设计**

## Rationale

*   原有的实验结果保存结构不够系统化，使得对比不同参数组合下的实验结果变得困难
*   目录名称中包含时间戳会使得路径过长，并且降低了实验结果的可读性和可比较性
*   需要一个更加结构化和标准化的目录结构，便于后续分析和对比不同实验条件下的结果
*   让实验的组织和检索更加直观，提高研究效率

## Implementation Details

*   重新设计了结果保存结构为多层级目录：
    1. 根目录使用时间戳命名：`results/experiments_YYYYMMDD_HHMMSS/`
    2. 实验总览文件 `experiment_overview.json` 和结果CSV文件保存在根目录
    3. 实验组合目录不再包含时间戳，采用参数组合命名：`dataset_Teacher_Student_hX_noiseN_smoothM/`
    4. 稳定性运行子目录标准化为：`runX_seed_YY/`
    5. 模型权重和绘图保存在相应的稳定性运行子目录中
*   修改了 `run_evaluation_experiments.py` 中的 `run_experiment()` 和 `main()` 函数，调整目录创建和路径引用逻辑
*   新增了直接生成 `experiment_overview.json` 的代码，确保实验配置信息完整保存