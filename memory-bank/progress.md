[2025-05-26 18:28:24] - 开始统计当前项目所有 Python 文件的代码行数。
2025-05-26 14:35:50 - 为模型抗噪音效果评估、数据去噪效果评估、Simi(Student,Teacher)评估以及增强Alpha调度器功能提供了详细的规范和伪代码。

This file tracks the project's progress using a task list format.
2025-05-26 13:10:00 - Log of updates made.

## Completed Tasks

* 建立了基本的项目结构，包括src/目录和数据处理模块
* 实现了核心模型接口，支持DLinear、PatchTST等多种时间序列模型
* 完成了RDT训练器实现，包括复合损失计算和Alpha调度
* 实现了数据加载和预处理流程，支持多种时间序列数据集
* 构建了评估框架，支持模型性能和鲁棒性测试
* 2025-05-26 17:44:00 - 决定暂不实现模型结构图可视化，因为这需要额外的库依赖（如 `graphviz`）且集成到训练流程中较为复杂。建议用户如果需要，可以手动使用 `torchviz` 或 `netron` 等工具生成。
* 2025-05-26 17:45:00 - 完成了详细的参数记录机制，包括实验元数据、模型架构参数、训练配置参数和数据处理参数的保存。
* 2025-05-26 14:43:44 - 完成了 `src/evaluator.py` 中学生-教师相似度评估功能的实现。
* 2025-05-26 14:44:21 - 完成了 `src/schedulers.py` 中 `AlphaScheduler` 基类及其子类 `update` 方法的文档更新。
* 2025-05-26 14:44:21 - 完成了 `src/trainer.py` 中 `RDT_Trainer` 在验证阶段收集预测结果并传递给 Alpha 调度器的实现。
* 2025-05-26 14:44:21 - 完成了 `src/data_handler.py` 中噪音注入和数据平滑/合成逻辑的实现。
* 2025-05-26 14:44:21 - 完成了 `src/config.py` 中噪音、平滑和相似度配置项的添加和修改。
* 2025-05-26 15:18:08 - 完成了训练过程中的输出日志增强和日志文件保存功能。
* 2025-05-26 15:24:26 - 完成了 `run_evaluation_experiments.py` 脚本的编写，实现了全面的模型评估实验功能。
* 2025-05-26 15:25:29 - 完成了 `run_quick_test_evaluation.py` 脚本的编写，用于快速验证代码和实验脚本的有效性。
* 2025-05-26 15:27:41 - 尝试运行 `run_quick_test_evaluation.py` 以验证 `ImportError` 修复。
* 2025-05-26 15:28:53 - 修复 `src/trainer.py` 中 `calculate_similarity` 的导入错误，将其更正为 `calculate_similarity_metrics`，并更新了相关函数调用。
* 2025-05-26 15:29:33 - 在 `src/utils.py` 中添加了 `save_results_to_csv` 函数。
* 2025-05-26 15:30:13 - 在 `src/utils.py` 中添加了 `save_plot` 函数。
* 2025-05-26 15:31:04 - 修复 `src/data_handler.py` 中 `load_and_preprocess_data` 函数的签名，使其接受 `dataset_path` 和 `logger` 参数，并替换了内部的 `print` 语句。
* 2025-05-26 15:32:32 - 修复 `run_evaluation_experiments.py` 中 `load_and_preprocess_data` 函数的解包错误，并更新 `get_model` 的调用。
* 2025-05-26 15:33:19 - 修复 `src/models.py` 中 `get_model` 函数的 `AttributeError`，使其接收 `Config` 实例并正确访问其属性，并删除了 `get_teacher_model` 和 `get_student_model` 函数。
* 2025-05-26 16:40:07 - 修复 `run_evaluation_experiments.py` 中 `StandardTrainer` 和 `RDT_Trainer` 的初始化参数。
* 2025-05-26 16:40:51 - 修复 `run_quick_test_evaluation.py` 中 `NameError`，导入了 `get_optimizer`, `get_loss_function` 和 `get_alpha_scheduler`。
* 2025-05-26 16:42:43 - 修复 `src/config.py` 中的 `RuntimeError`，添加 `update_model_configs` 方法并在 `run_quick_test_evaluation.py` 和 `run_evaluation_experiments.py` 中调用。
* 2025-05-26 16:43:55 - 修复 `src/trainer.py` 中 `evaluate_model` 函数的 `TypeError`，使其 `__init__` 方法接受 `scaler` 参数，并在 `_validate_epoch` 中正确调用 `evaluate_model`。
* 2025-05-26 16:44:39 - 修复 `run_evaluation_experiments.py` 中 `StandardTrainer` 和 `RDT_Trainer` 的初始化参数，传入 `scaler`。
* 2025-05-26 16:46:09 - 修复 `run_quick_test_evaluation.py` 和 `run_evaluation_experiments.py` 中 `evaluate_model` 函数的调用，传入正确的参数并处理教师模型预测。
* 2025-05-26 16:50:58 - 修复 `run_evaluation_experiments.py` 中 `evaluate_model` 函数的调用，传入正确的参数并处理教师模型预测。
* 2025-05-26 17:00:22 - 在 `src/evaluator.py` 的 `predict` 函数中添加了类型检查，以确保 `model` 参数是 `torch.nn.Module` 的实例。
* 2025-05-26 17:01:38 - 完成了 `src/trainer.py` 中 `_validate_epoch` 函数的修改，以确保 `evaluate_model` 接收到正确的 `model` 对象和 `dataloader`。
* 2025-05-26 17:04:39 - 完成对 `src/evaluator.py` 和 `src/trainer.py` 中所有必要代码的修复。
* 2025-05-26 17:05:30 - 完成对 `run_quick_test_evaluation.py` 中 `StandardTrainer` 实例化参数的修复。
* 2025-05-26 17:07:55 - 完成对 `run_quick_test_evaluation.py` 中 `evaluate_model` 调用参数的修复。
* 2025-05-26 17:08:56 - 完成对 `src/trainer.py` 中 `train` 函数的修复，正确处理 `_validate_epoch` 返回的元组。
* 2025-05-26 17:10:13 - 完成对 `src/trainer.py` 中 `_validate_epoch` 返回值的修复。
* 2025-05-26 17:11:09 - 完成对 `run_quick_test_evaluation.py` 中 `evaluate_model` 调用参数的修复，移除 `logger` 参数。
* 2025-05-26 17:12:05 - 完成对 `run_quick_test_evaluation.py` 中目录创建的修复。
* 2025-05-26 17:13:14 - 完成对 `run_quick_test_evaluation.py` 中 `results` 和 `similarity_results` 字典初始化的修复。
* 2025-05-26 17:16:47 - 完成对 `src/trainer.py` 中 `RDT_Trainer._validate_epoch` 调用 `evaluate_model` 参数的修复。
* 2025-05-26 17:18:49 - 完成对 `run_quick_test_evaluation.py` 中 `RDT_Trainer` 实例化参数的修复。
* 2025-05-26 17:20:21 - 完成对 `run_quick_test_evaluation.py` 中 `TaskOnly` 模型的 `RDT_Trainer` 实例化参数的修复。
* 2025-05-26 17:21:15 - 完成对 `src/trainer.py` 中 `RDT_Trainer._validate_epoch` 调用 `calculate_similarity_metrics` 参数名的修复。
* 2025-05-26 17:23:15 - 完成对 `src/trainer.py` 中 `RDT_Trainer._validate_epoch` 返回值的修复。
* 2025-05-26 17:24:46 - 完成对 `src/trainer.py` 中 `RDT_Trainer._validate_epoch` 相似度指标处理的修复。
* 2025-05-26 17:30:03 - 完成对 `run_evaluation_experiments.py` 的所有必要修复。
* 2025-05-26 17:40:00 - 完成了核心训练指标（训练损失、验证损失、学习率、梯度范数和评估指标）的可视化功能。
* 2025-05-26 17:41:00 - 完成了模型权重和偏置分布的可视化功能。
* 2025-05-26 17:43:00 - 完成了增强预测结果与真实值对比的可视化功能（包括残差分析和误差分布图）。
* [2025-05-26 18:07:26] - 完成 `src/utils.py` 中 `plot_predictions` 函数的 `title` 和 `save_path` 未定义警告修复。
* [2025-05-26 18:38:32] - 为 `run_evaluation_experiments.py` 的训练过程增加了进度条。
* [2025-05-26 21:22:00] - 修复 `src/data_handler.py` 中 `load_and_preprocess_data` 函数在 `denoising_smoothing` 实验类型下测试集未平滑的问题，并增加了 `src/config.py` 中的 `SMOOTHING_APPLY_TEST` 配置项。
* [2025-05-26 21:22:00] - 修复 `src/evaluator.py` 中 `evaluate_model` 和 `evaluate_robustness` 函数使用 `print` 而非 `logger` 的问题，并更新了 `run_evaluation_experiments.py` 中对这些函数的调用。
* [2025-05-26 21:22:00] - 修复 `src/trainer.py` 中 `StandardTrainer` 的 `_validate_epoch` 函数 `evaluate_model` 返回值解包不正确的问题。
* [2025-05-26 21:22:00] - 修复 `src/trainer.py` 中 `EarlyStopping` 的 `trace_func` 默认使用 `print` 的问题，改为使用 `logging.info`。
* [2025-05-26 21:22:00] - 修复 `src/trainer.py` 中 `BaseTrainer` 的 `self.scaler` 重复赋值问题。
* [2025-05-26 21:22:00] - 修复 `src/trainer.py` 中 `RDT_Trainer._validate_epoch` 函数在调用 `evaluate_model` 时，`teacher_predictions_original` 参数传入的是 `scaled` 预测结果的问题，现在会传入逆变换后的原始尺度预测结果。
* 2025-05-26 21:27:30 - 完成 `run_evaluation_experiments.py` 中噪音注入和平滑处理的配置修改。
* 2025-05-27 00:15:10 - 完成 `README.md` 文档更新。
* [2025-05-27 01:21:59] - 完成 `run_evaluation_experiments.py` 脚本修改任务，使其在保存模型和结果时，使用新的命名格式。
* [2025-05-27 01:36:02] - 进一步优化 `run_evaluation_experiments.py`，在结果文件夹命名中包含随机种子和运行索引，以提高结果复现性。
* [2025-05-27 02:07:30] - 完成对 `run_evaluation_experiments.py` 结果保存结构的优化，实现了统一规范的多层级目录结构：1) 时间戳根目录 2) 实验总览文件和CSV结果文件保存在根目录 3) 参数组合目录名不含时间戳 4) 稳定性运行子目录格式标准化为 `runX_seed_YY`。
* [2025-05-30 18:27:35] - 完成 `src/data_handler.py` 中 `time_features()` 和 `cyclic_time_features()` 函数的修改，将直接访问时间属性改为使用 `.dt` 访问器。
* [2025-05-30 18:41:29] - 完成 `src/data_handler.py` 中 `load_and_preprocess_data` 函数的修改，使其返回 `n_features`。
* [2025-05-30 18:41:29] - 完成 `run_quick_test_evaluation.py` 中 `run_experiment` 函数的修改，以捕获 `n_features` 并更新模型配置。
* [2025-05-30 18:41:29] - 完成 `run_evaluation_experiments.py` 中 `run_experiment` 函数的修改，以捕获 `n_features` 并更新模型配置。
* [2025-05-30 18:47:47] - 完成了对 `src/config.py`、`src/data_handler.py`、`run_quick_test_evaluation.py`、`run_evaluation_experiments.py` 和 `src/models.py` 的修改，以统一处理特征数量 `n_features`。
* [2025-05-30 19:00:00] - 完成 DLinear 模型维度不匹配错误的修复 (修改 trainer.py 和 evaluator.py)。
* [2025-05-30 19:12:00] - 完成 `src/trainer.py` 和 `src/evaluator.py` 的修改，以解决损失计算和评估中的维度不匹配问题。
* [2025-05-30 19:23:00] - Completed: Modified `src/trainer.py` and `src/evaluator.py` to fix dimension mismatch issues related to `inverse_transform` and ensure correct handling of target columns for loss and metric calculations.
* [2025-05-30 21:33:00] - 完成代码修改，允许模型输入包含额外的协变量。更新了 `src/config.py`, `src/data_handler.py`, `run_evaluation_experiments.py`, 和 `run_quick_test_evaluation.py`。
* [2025-05-30 22:11:00] - 完成了在模型训练开始前打印 CUDA 运行状态的功能。修改了 `src/trainer.py`，`run_quick_test_evaluation.py` 和 `run_evaluation_experiments.py`。
* [2025-05-30 22:32:51] - Completed: Ensured default dropout (0.3) and head_dropout (0.0) for PatchTST models by modifying `src/models.py` and `src/config.py`.
* [2025-05-30 22:56:00] - 完成时间处理逻辑优化，以支持“分钟”级别，并同步更新到相关实验脚本和配置文件。具体包括更新 `src/data_handler.py` 中的时间特征函数，在 `src/config.py` 中引入 `DATASET_TIME_FREQ_MAP`，以及修改实验脚本以传递 `time_freq` 参数。
* [2025-05-31 00:43:15] - 完成对 `src/evaluator.py` 的修改：移除 `evaluate_model` 函数中未使用的 `plots_dir` 参数，调整绘图代码块位置并更新其目录参数。
* [2025-05-31 01:03:59] - 完成 `error_cos_similarity` 指标的实现。在 `src/evaluator.py` 中添加了 `calculate_error_cosine_similarity` 函数，并修改了 `evaluate_model` 以集成此新指标。确认了实验脚本 (`run_evaluation_experiments.py`, `run_quick_test_evaluation.py`) 的现有CSV保存逻辑将自动处理新指标。
* [2025-05-31 01:23:25] - 完成 `run_evaluation_experiments.py` 中 `evaluate_model` 函数调用时 `TypeError` 的修复。
* [2025-05-31 01:46:51] - 完成 `TypeError: evaluate_model() got an unexpected keyword argument 'plots_dir'` 错误的修复。
* [2025-05-31 02:08:45] - 完成优化图表绘制时机和位置。在 `src/trainer.py` 中移除了验证阶段的 `evaluate_model` 调用和训练结束时的图表绘制。在 `run_evaluation_experiments.py` 中，在每个模型训练完成后加载最佳模型，并统一调用 `evaluate_model` 进行最终评估和图表绘制。
* [2025-05-31 02:18:32] - 修复 `run_metrics.json` 中 `NaN` 值导致文件无法打开的问题。
* [2025-05-31 02:15:10] - 修复 `run_evaluation_experiments.py` 中 `save_plot` 未定义错误。
* [2025-05-31 02:38:00] - Created pseudocode for `run_evaluation_alpha.py`
* [2025-05-31 02:42:55] - Created `run_evaluation_alpha.py` script.
* [2025-05-31 02:52:02] - 诊断并修复 `val_loader` 为空导致的 `ZeroDivisionError`。通过调整 `src/config.py` 中的 `VAL_SPLIT_RATIO` 和 `TEST_SPLIT_RATIO` 来确保验证集有足够的样本。
* [2025-05-31 10:17:30] - 完成对 `run_evaluation_experiments.py` 文件的内存优化，解决内存泄漏问题。添加了 `gc` 和 `psutil` 模块，实现了 `clean_memory` 和 `clean_between_experiments` 函数，在关键点添加了内存清理调用，优化了对象生命周期管理，并改进了 `convert_floats` 函数。
* [2025-05-31 13:10:00] - 修复了 `run_evaluation_alpha.py` 中固定alpha模型未进行数据反归一化处理的问题。问题根源在于创建固定alpha模型的配置对象时，没有设置 `N_FEATURES` 属性，导致反归一化失败，使用归一化后的数据计算指标。通过添加 `current_fixed_alpha_config.N_FEATURES = config.N_FEATURES` 解决了这个问题。
* [2025-05-31 13:14:00] - 修复了 `run_evaluation_alpha.py` 中的 `KeyError: 'dataset'` 错误。问题出在 `plot_fixed_alpha_evaluation` 函数中，当尝试访问 `fixed_alpha_sim_df` 的 'dataset' 列时。通过添加额外的检查，确保在尝试访问列之前，检查数据框是否为空或是否有必要的列，避免了程序崩溃。
* [2025-06-02 16:07:00] - 修复 `run_evaluation_experiments.py` 中的 `KeyError: 'experiment_type'` 错误。
* [2025-06-03 16:22:35] - 完成创建 `run_custom_experiment.py` 脚本，实现了多模型和多预测窗口的实验循环，并集成了数据加载、模型训练、评估和结果保存功能。
* [2025-06-03 16:28:40] - 完成修改 `run_custom_experiment.py` 文件，将 `setup_logging` 函数的调用参数从 `config` 修改为 `config.LOG_FILE_PATH` 和 `config.LOG_LEVEL`。
* [2025-06-03 16:32:48] - 完成修改 `run_custom_experiment.py` 文件，将训练模型部分的代码修改为 `trained_model, _ = trainer.train()`，并将评估模型部分的代码修改为直接使用 `trained_model`，移除了冗余的 `load_state_dict` 和 `model.to(config.DEVICE)`。
* [2025-06-03 16:34:34] - 完成修复 `run_custom_experiment.py` 中 `plot_weights_biases_distribution` 函数调用时 `best_model` 未定义的错误，将其修改为 `trained_model`。
* [2025-06-03 16:51:16] - Completed: Created `run_evaluation_no_plots.py` script by removing plotting functionality from `run_quick_test_evaluation.py`.
* [2025-06-03 16:59:16] - Completed: Modified [`run_evaluation_no_plots.py`](run_evaluation_no_plots.py) to correctly handle `teacher_model_name = None` scenarios.
* [2025-06-03 17:07:24] - 完成对 [`src/models.py`](src/models.py:1) 中 `MLPModel` ([`src/models.py:185`](src/models.py:185)), `RNNModel` ([`src/models.py:213`](src/models.py:213)), 和 `LSTMModel` ([`src/models.py:246`](src/models.py:246)) `forward` 方法的修改，以解决输入维度不匹配问题。
* [2025-06-03 17:30:23] - 完成对 [`src/models.py`](src/models.py:1) 的修改，以确保自定义模型 (`MLPModel`, `RNNModel`, `LSTMModel`) 输出正确数量的特征 (即 `len(config.TARGET_COLS)`)。
* [2025-06-03 17:33:48] - 完成对 `src/evaluator.py` 中 `evaluate_model` 函数的修改，以正确处理模型输出特征数多于目标列的情况。
* [2025-06-03 23:04:06] - 完成在 `src/config.py` 和 `src/evaluator.py` 中实现默认不绘制详细评估图表的功能。

* [2025-06-15 02:03:29] - 完成对 `run_evaluation_no_plots.py` 的修改，通过引入 `results_dir` 参数解决了模型保存路径冲突和覆盖的问题。
* [2025-06-15 01:47:49] - 完成了在 `src/config.py` 中添加 WAPE 和 MAPE 评估指标的任务。
* [2025-06-15 02:25:09] - 完成对 `run_evaluation_no_plots.py` 的修改。在 'denoising_smoothing' 实验中，为训练、验证和测试集明确启用了平滑处理，以确保数据处理流程的一致性。
## Current Tasks

* 初始化Memory Bank以记录项目架构和设计决策
* 分析现有代码库的架构和组件关系
* 理解RDT方法的核心实现细节

## Next Steps

* 探索更多Alpha调度策略的实现
* 考虑集成更多种类的时间序列模型作为教师或学生
* 优化数据预处理流程，特别是针对不同类型的时间序列数据
* 增强可视化能力，提供更直观的实验结果展示
* 改进文档，添加更详细的使用说明和API参考
- 2025-05-26 下午2:45:18 - 开始编写新功能文档。
- 2025-05-26 下午2:45:49 - 完成新功能文档 `docs/new_features.md` 的编写。
* [2025-06-15 02:16:51] - 完成对 `run_evaluation_no_plots.py` 的修复，确保在 'denoising_smoothing' 实验中平滑化功能能够根据配置正确生效。
* [2025-06-15 02:47:06] - [Debugging Task Status Update] 完成了对 `smooth_data` 函数 `ValueError` 的修复。
* [2025-06-15 19:42:04] - 完成对 `run_evaluation_no_plots.py` 的修改，通过引入 `results_dir` 参数解决了模型保存路径冲突和覆盖的问题。
* [2025-06-15 19:58:22] - 完成对 `run_evaluation_no_plots.py` 的修改，通过实现详细的模型命名约定，解决了因通用文件名导致模型相互覆盖的问题。
* [2025-06-15 22:59:48] - 完成了将 `run_experiment` 函数重构到 `src/run_experiment.py` 的任务。
[2025-06-15 23:57:28] - COMPLETED: Created new documentation file 'covariate_handling_explanation.md' detailing PatchTST covariate handling.