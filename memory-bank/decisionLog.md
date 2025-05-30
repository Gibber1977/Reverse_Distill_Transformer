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

---

## Decision

*   **时间特征编码策略：优先采用周期性编码（正弦/余弦变换）**

## Rationale

*   时间序列数据中的小时、星期几、月份、一年中的天数等特征具有固有的周期性。直接使用原始数值无法有效捕捉这种周期性关系，可能导致模型难以学习到准确的模式。
*   周期性编码（正弦/余弦变换）能够将周期性特征映射到二维空间，使得周期开始和结束的点在特征空间中是相邻的，从而有效捕捉其周期性。这避免了原始数值大小带来的偏置，并以连续且有意义的方式呈现给模型。
*   独热编码虽然可以明确区分类别，但对于周期性特征不如正弦/余弦变换有效，且容易导致维度爆炸，增加模型复杂度和计算成本。

## Implementation Details

*   当前代码中 `src/data_handler.py` 已实现 `cyclic_time_features()` 函数，该函数通过正弦/余弦变换对小时、星期几、一年中的天数、月份和一年中的周数进行周期性编码，这符合时间序列预测的最佳实践。
*   建议保留并优先使用 `cyclic_time_features()` 作为时间特征编码的主要方式。
*   对于非周期性特征（如年份），应将其作为数值特征进行标准化，而不是进行周期性编码。
*   如果 `linear` 编码类型（`time_features()`）仍需保留，应在文档中明确其局限性，并强调 `cyclic` 编码的优势。
---
### Decision (Code)
[2025-05-30 18:48:03] - 统一模型特征数量处理方式

**Rationale:**
为了更灵活地处理不同数据集的特征数量，并简化模型配置，决定将特征数量的确定和传递方式进行统一。原先在 `Config` 类中，每个模型的配置字典都独立定义了 `n_series`，这可能导致在数据集特征变化时需要多处修改。新的方式通过在 `data_handler.py` 中动态获取特征数，并将其存储在 `Config` 类的一个新属性 `N_FEATURES` 中，然后在 `models.py` 的 `get_model` 函数中统一使用此属性来配置各个模型。

**Details:**
- **[`src/config.py`](src/config.py)**:
    - 从 `Config` 类中所有模型配置字典（`TEACHER_CONFIG`, `STUDENT_CONFIG`, `NLINEAR_CONFIG`, `MLP_CONFIG`, `RNN_CONFIG`, `LSTM_CONFIG`, `AUTOFORMER_CONFIG`, `INFORMER_CONFIG`, `FEDFORMER_CONFIG`）中删除了 `n_series` 的初始化。
    - 在 `Config` 类中添加了 `self.N_FEATURES = None`。
    - 在 `update_model_configs()` 方法中删除了所有更新 `n_series` 的行。
- **[`src/data_handler.py`](src/data_handler.py)**:
    - 修改 [`load_and_preprocess_data()`](src/data_handler.py:293-446) 函数的返回语句为 `return train_loader, val_loader, test_loader, scaler, df_target.shape[1]`，直接返回特征数量。
- **[`run_quick_test_evaluation.py`](run_quick_test_evaluation.py) 和 [`run_evaluation_experiments.py`](run_evaluation_experiments.py)**:
    - 修改了调用 `load_and_preprocess_data` 的地方，以捕获新的返回值 `n_features`。
    - 添加了 `config.N_FEATURES = n_features`。
    - 删除了之前手动更新 `config.TEACHER_CONFIG['n_series']` 等的所有行。
- **[`src/models.py`](src/models.py)**:
    - 在 `get_model` 函数中，将所有模型的 `n_series` 参数来源从 `config.MODEL_CONFIG['n_series']` 修改为 `config.N_FEATURES`。
    - 对于 `RNN` 和 `LSTM` 模型，将其 `output_size` 参数也修改为 `config.N_FEATURES`。
---
### Decision (Code)
[2025-05-30 19:00:00] - 调整模型输入以兼容 DLinear

**Rationale:**
DLinear 模型期望输入分为 `insample_y` (目标序列的历史值) 和 `X_df` (外生特征)。为了解决维度不匹配错误并确保与 DLinear 及其他类似接口的模型兼容，需要修改数据传递给模型的方式。

**Details:**
- 在 `src/trainer.py` 的 `StandardTrainer` 和 `RDT_Trainer` 的 `_train_epoch` 和 `_validate_epoch` 方法中，将 `batch_x` 拆分为 `insample_y` 和 `X_df_batch`，并更新 `input_dict`。
- 在 `src/evaluator.py` 的 `predict` 函数中（影响 `evaluate_model`），同样对 `batch_x` 进行拆分并更新 `input_dict`。
- `config_obj.TARGET_COLS` 用于确定目标列的数量，从而正确拆分 `insample_y` 和 `X_df_batch`。
---
### Decision (Code)
[2025-05-30 19:12:00] - 解决损失计算和评估中的维度不匹配问题

**Rationale:**
在训练和评估过程中，模型的输出 (`outputs` 或 `student_outputs`) 和目标 (`target_y` 或 `batch_y_true`) 之间可能存在维度不匹配，特别是当 `target_y` 包含的特征列多于 `config.TARGET_COLS` 中定义的目标列时。为了确保损失函数和评估指标仅基于预期的目标列进行计算，需要对目标张量进行切片。

**Details:**
- **[`src/trainer.py`](src/trainer.py)**:
    - 在 `StandardTrainer` 的 `_train_epoch` 和 `_validate_epoch` 方法中，在计算损失之前，将 `batch_y` 切片为 `target_y_for_loss = batch_y[:, :, :len(self.config.TARGET_COLS)]`。
    - 在 `RDT_Trainer` 的 `_train_epoch` 和 `_validate_epoch` 方法中，在计算任务损失之前，将 `batch_y_true` 切片为 `target_y_for_loss = batch_y_true[:, :, :len(self.config.TARGET_COLS)]`。
- **[`src/evaluator.py`](src/evaluator.py)**:
    - 在 `predict` 函数中，将 `batch_y_device` 切片为 `target_y_to_append = batch_y_device[:, :, :len(config_obj.TARGET_COLS)]` 后再添加到 `all_trues` 列表。
    - 在 `evaluate_model` 函数中，在对 `true_values_original` 和 `predictions_original` 进行逆变换和重塑之后，但在计算指标之前，添加了以下代码以确保只使用目标列：
      ```python
      true_values_original = true_values_original[:, :, :len(config_obj.TARGET_COLS)]
      predictions_original = predictions_original[:, :, :len(config_obj.TARGET_COLS)]
      ```
---
### Decision (Code)
[2025-05-30 19:23:00] - Refined inverse_transform logic for predictions and truth values

**Rationale:**
To address dimensionality mismatches when using `scaler.inverse_transform`. The scaler is typically fit on `N_FEATURES`, but model predictions (`predictions_scaled`) and target values (`true_values_scaled`, `teacher_predictions_scaled`) often only contain `len(TARGET_COLS)` features. Directly applying `inverse_transform` would fail or produce incorrect results. The solution involves creating a dummy array of shape `(num_samples * horizon, N_FEATURES)`, populating the target column(s) with the scaled predictions/values, applying `inverse_transform` to this dummy array, and then extracting only the target column(s) from the result. This ensures correct inverse transformation for evaluation and similarity calculations.

**Details:**
- **[`src/evaluator.py`](src/evaluator.py)**:
    - Modified `evaluate_model` to use a helper function `_inverse_transform_target_cols` for both `predictions_scaled` and `true_values_scaled`. This helper implements the "dummy array" strategy.
    - Ensured subsequent `reshape` operations use `len(config.TARGET_COLS)`.
- **[`src/trainer.py`](src/trainer.py)**:
    - Modified `RDT_Trainer._validate_epoch` to apply the same "dummy array" strategy for `teacher_preds_all` before passing `teacher_predictions_original` to `evaluate_model`. This ensures consistency.
    - Confirmed that target slicing for loss calculation (`target_y_for_loss = ...[:, :, :len(self.config.TARGET_COLS)]`) was already correctly implemented in `StandardTrainer` and `RDT_Trainer` methods as per prior decisions.
---
### Decision (Code)
[2025-05-30 21:32:00] - 允许模型输入包含额外的协变量

**Rationale:**
为了增强模型的预测能力，允许在模型输入中包含原始数据中的其他协变量。这使得模型可以利用除目标变量和时间特征之外的更多信息进行学习。

**Details:**
- **[`src/config.py`](src/config.py:18)**:
    - 在 `Config` 类的 `__init__` 方法中，添加了 `self.EXOGENOUS_COLS = []`。这是一个列表，用于指定要从原始数据中选择作为额外协变量的列名。如果为空，则不使用额外协变量。
- **[`src/data_handler.py`](src/data_handler.py)**:
    - 在 [`load_and_preprocess_data()`](src/data_handler.py:293-447) 函数中：
        - 在选择目标列 `df_target` 之后，添加了逻辑以根据 `cfg.EXOGENOUS_COLS` 从原始数据帧 `df` 中选择协变量列，并存入 `df_exog`。如果指定的列不存在，会记录警告。
        - 修改了数据合并逻辑：原先只合并 `df_target` 和 `df_time_features`。现在，按顺序合并 `df_target`、`df_exog` (如果存在且不为空) 和 `df_time_features` (如果存在) 到一个新的 `df_processed` DataFrame 中。目标列始终位于最前面。
        - 后续所有对 `df_target` 的引用（如缺失值处理、数据划分、标准化、噪音注入等）都已更改为 `df_processed`。
        - 函数末尾返回的特征数量更新为 `df_processed.shape[1]`。
- **[`run_evaluation_experiments.py`](run_evaluation_experiments.py:130-136) 和 [`run_quick_test_evaluation.py`](run_quick_test_evaluation.py:86-92)**:
    - 在这两个实验脚本的 `run_experiment` 函数内部，在创建 `config = Config()` 实例后，添加了根据 `dataset_name` 设置 `config.EXOGENOUS_COLS` 的逻辑。例如，为 `exchange_rate` 数据集指定了协变量列 `['0', '1', '2', '3', '4', '5', '6']`，而其他数据集则设置为空列表。
---
## Decision
* [2025-05-30 22:08:00] - 在训练开始前记录模型运行设备

## Rationale
* 为了在实验日志中清晰地标示模型是在 CUDA 还是 CPU 上运行，便于调试和性能分析。

## Implementation Details
* 在 `src/trainer.py` 的 `BaseTrainer` (或其相关子类) 的 `train` 方法的起始位置，检查 `self.model` 的参数所在的设备。
* 使用 `logger.info` 打印设备信息。
* 确保模型在训练脚本中被正确地移动到目标设备，并在日志中反映实际使用的设备。

---
### Decision (Code)
[2025-05-30 22:11:00] - 实现训练前打印CUDA运行状态

**Rationale:**
根据用户请求，在模型训练开始前明确记录模型将使用的计算设备（CPU 或 CUDA），并确认 CUDA 的可用性。这有助于调试和验证实验是否在预期的硬件上运行。

**Details:**
- **[`src/trainer.py`](src/trainer.py:96-109)**:
    - 在 `BaseTrainer` 的 `train` 方法的开头，添加了逻辑来检查 `self.model` 的参数所在的设备。
    - 使用 `logging.info` 打印模型设备信息，并明确指出是在 CUDA 还是 CPU 上运行。
    - 使用 `try-except` 块来处理潜在的错误。
- **[`run_quick_test_evaluation.py`](run_quick_test_evaluation.py:118-132)** 和 **[`run_evaluation_experiments.py`](run_evaluation_experiments.py:172-186)**:
    - 在 `run_experiment` 函数中，获取模型之前，添加了代码来检查 `config.DEVICE` 的设置和 `torch.cuda.is_available()` 的状态。
    - 根据检查结果，更新 `config.DEVICE` 为实际使用的设备 (`"cpu"` 或 CUDA 设备字符串)。
    - 使用 `logger.info` 记录配置请求的设备、CUDA可用性以及最终实验将在哪个设备上运行。
    - 确保在获取模型后，使用 `.to(device)` 将模型显式移动到已确定的设备上。
---
### Decision (Code)
[2025-05-30 22:32:11] - Ensured default dropout values for PatchTST models.

**Rationale:**
To maintain consistent model behavior and training stability, specific default values for `dropout` (0.3) and `head_dropout` (0.0) are enforced for PatchTST models if not explicitly set in configurations. This aligns with common practices and provides sensible defaults.

**Details:**
- Modified [`src/models.py`](src/models.py:67) in the `get_model` function to default `dropout` to 0.3 and `head_dropout` to 0.0 for PatchTST if not specified in the configuration dictionary (`cfg`). A copy of `cfg` is used to avoid modifying the original.
- Modified [`src/config.py`](src/config.py:40) to explicitly set `dropout: 0.3` and `head_dropout: 0.0` in the `STUDENT_CONFIG` (which defaults to PatchTST) for clarity and as a documented default.
---
### Decision (Code)
[2025-05-30 22:56:00] - 优化时间处理逻辑以支持分钟级特征，并引入数据集到时间频率的映射。

**Rationale:**
为了使模型能够处理不同时间频率（特别是分钟级）的时间序列数据，需要对数据处理模块和配置进行相应调整。将数据集到时间频率的映射集中管理可以提高代码的可维护性和灵活性。

**Details:**
- **[`src/data_handler.py`](src/data_handler.py)**:
    - 更新 `time_features` 和 `cyclic_time_features` 函数以支持分钟级时间特征。具体决策包括：
        - 分钟级特征将包括：分钟本身 (`minute_of_hour`)。
        - 保留原有的小时 (`hour_of_day`)、日 (`day_of_week`, `day_of_month`, `day_of_year`)、周 (`week_of_year`) 和月 (`month_of_year`) 特征，以便模型能够捕捉不同时间尺度上的模式。
    - 修改 `load_and_preprocess_data` 函数，使其能够接收 `time_freq` 参数，并将其传递给 `time_features` 或 `cyclic_time_features` 函数。
- **[`src/config.py`](src/config.py)**:
    - 在 `Config` 类中引入 `DATASET_TIME_FREQ_MAP` 字典，用于存储数据集文件名到其对应时间频率的映射。
        - `data\ETT-small\ETTh1.csv`: 'h'
        - `data\ETT-small\ETTh2.csv`: 'h'
        - `data\ETT-small\ETTm1.csv`: 'min'
        - `data\ETT-small\ETTm2.csv`: 'min'
        - `data\exchange_rate.csv`: 'd'
        - `data\national_illness.csv`: 'w'
        - `data\weather.csv`: 'min'
    - 添加 `TIME_FREQ` 属性到 `Config` 类，该属性将在运行时根据当前处理的数据集从 `DATASET_TIME_FREQ_MAP` 动态设定。
- **[`run_quick_test_evaluation.py`](run_quick_test_evaluation.py) 和 [`run_evaluation_experiments.py`](run_evaluation_experiments.py)**:
    - 在实验脚本中增加逻辑，在加载数据前，根据当前数据集的文件名从 `config.DATASET_TIME_FREQ_MAP` 查询对应的时间频率。
    - 将查询到的时间频率 (`time_freq`) 传递给 `load_and_preprocess_data` 函数。
---
### Decision (Code)
[2025-05-30 23:00:00] - 实现对分钟级时间特征的支持

**Rationale:**
为了使模型能够处理更高频率的时间序列数据（例如 ETTm1, ETTm2, weather 数据集），需要扩展时间特征工程以包含分钟级信息。这包括线性特征（如 `minute_of_hour`）和周期性特征（`sin(minute_of_hour)`, `cos(minute_of_hour)`）。同时，需要一种机制来为不同的数据集指定其固有的时间频率。

**Details:**
- **[`src/config.py`](src/config.py:1):**
    - 添加了 `DATASET_TIME_FREQ_MAP` 字典，将数据集文件名映射到其时间频率（例如 `'ETTm1.csv': 'min'`）。
    - `TIME_FREQ` 属性现在作为默认值，实际频率在实验脚本中根据 `DATASET_TIME_FREQ_MAP` 动态设置。
- **[`src/data_handler.py`](src/data_handler.py:1):**
    - `time_features()` 函数：当 `freq='min'` 时，添加 `dates.dt.minute` 特征，并保留小时、星期几等其他相关特征。归一化逻辑保持一致。
    - `cyclic_time_features()` 函数：当 `freq='min'` 时，添加分钟的 `sin/cos` 周期特征，并保留其他相关周期特征。
    - `load_and_preprocess_data()` 函数：
        - 修改函数签名以接受新的 `time_freq` 参数: `def load_and_preprocess_data(dataset_path, cfg, logger, time_freq):`
        - 在函数内部，将传入的 `time_freq` 参数传递给 `time_features` 和 `cyclic_time_features` 函数的 `freq` 参数。
- **[`run_quick_test_evaluation.py`](run_quick_test_evaluation.py:1) 和 [`run_evaluation_experiments.py`](run_evaluation_experiments.py:1):**
    - 在 `run_experiment` 函数中，在调用 `load_and_preprocess_data` 之前：
        - 从 `dataset_path` 提取文件名。
        - 使用文件名从 `config.DATASET_TIME_FREQ_MAP` 查找时间频率。
        - 将查找到的时间频率赋值给 `config.TIME_FREQ` (用于日志和可能的其他用途)。
        - 将查找到的时间频率作为 `time_freq` 参数传递给 `load_and_preprocess_data` 函数调用。