# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions.
2025-05-26 13:09:00 - Log of updates made.

## Current Focus

* 为现有训练模型代码增加模型抗噪音效果评估功能。
* 为现有训练模型代码增加数据去噪效果评估功能。
* 为训练结果增加 Simi(Student,Teacher) 效果评估。
* 增强 `get_alpha_scheduler`，使其能够基于验证集训练信息和模型相似度调整 α 权重。

## Recent Changes

* 2025-05-26 13:09:00 - 创建了Memory Bank基础结构
* 2025-05-26 13:09:00 - 分析了主要模块(models.py, trainer.py, data_handler.py)的实现逻辑
* 2025-05-26 14:35:24 - 更新了 `productContext.md` 以反映新功能需求。

## Open Questions/Issues

* 当前实现支持哪些Alpha调度策略？需要检查schedulers.py
* 对于不同的时间序列数据集，可能需要哪些特定的预处理步骤？
* 如何优化教师-学生模型组合以获得最佳性能？
* 项目的测试覆盖度如何？是否需要添加更多单元测试或集成测试？
* 如何在 `data_handler.py` 中实现噪音注入和数据平滑处理，同时保持模块化和可配置性？
* 如何在 `evaluator.py` 中集成新的评估指标和比较逻辑？
* 如何在 `trainer.py` 中传递验证集信息给 `alpha_scheduler`？
* 2025-05-26 14:43:55 - 完成了 `src/evaluator.py` 中学生-教师相似度评估功能的实现。
* 2025-05-26 14:44:29 - 完成了 `src/schedulers.py` 中 `AlphaScheduler` 基类及其子类 `update` 方法的文档更新。
* 2025-05-26 14:44:29 - 完成了 `src/trainer.py` 中 `RDT_Trainer` 在验证阶段收集预测结果并传递给 Alpha 调度器的实现。
* 2025-05-26 14:44:29 - 完成了 `src/data_handler.py` 中噪音注入和数据平滑/合成逻辑的实现。
* 2025-05-26 14:44:29 - 完成了 `src/config.py` 中噪音、平滑和相似度配置项的添加和修改。
* 2025-05-26 15:17:53 - 增强了训练过程中的输出日志功能，包括每个 epoch 的训练损失、验证损失、评估指标（包括 Simi(Student,Teacher)）、Alpha 权重动态调整信息、噪音注入和平滑处理的配置信息。日志现在保存到 `log` 文件夹下。
* 2025-05-26 15:24:19 - 创建了 `run_evaluation_experiments.py` 脚本，用于执行全面的模型评估实验，包括噪音注入、去噪平滑和相似度评估。
* 2025-05-26 15:25:23 - 创建了 `run_quick_test_evaluation.py` 脚本，用于快速验证代码和实验脚本的有效性。
* 2025-05-26 15:28:32 - 发现 `run_quick_test_evaluation.py` 仍然存在 `ImportError: cannot import name 'Config' from 'src.config'`，同时发现 `src/trainer.py` 中存在 `ImportError: cannot import name 'calculate_similarity' from 'src.evaluator'`，应为 `calculate_similarity_metrics`。
* 2025-05-26 15:29:15 - 发现 `run_quick_test_evaluation.py` 存在 `ImportError: cannot import name 'save_results_to_csv' from 'src.utils'`。
* 2025-05-26 15:29:55 - 发现 `run_quick_test_evaluation.py` 存在 `ImportError: cannot import name 'save_plot' from 'src.utils'`。
* 2025-05-26 15:30:34 - 发现 `run_quick_test_evaluation.py` 存在 `TypeError: load_and_preprocess_data() takes 1 positional argument but 3 were given`，与 `load_and_preprocess_data` 函数的调用有关。
* 2025-05-26 15:32:04 - 发现 `run_quick_test_evaluation.py` 存在 `ValueError: not enough values to unpack (expected 5, got 4)`，因为 `load_and_preprocess_data` 函数不返回 `data_info`。
* 2025-05-26 15:32:56 - 发现 `run_quick_test_evaluation.py` 存在 `AttributeError: module 'src.config' has no attribute 'LOOKBACK_WINDOW'`，发生在 `src/models.py` 中，因为 `config` 模块被导入但未实例化 `Config` 类。
* 2025-05-26 16:39:16 - 发现 `run_quick_test_evaluation.py` 存在 `TypeError: StandardTrainer.__init__() got an unexpected keyword argument 'test_loader'`，与 `StandardTrainer` 类的初始化有关。
* 2025-05-26 16:39:29 - 发现 `StandardTrainer` 的 `__init__` 方法不接受 `test_loader` 参数，导致 `TypeError`。
* 2025-05-26 16:40:36 - 发现 `run_quick_test_evaluation.py` 存在 `NameError: name 'get_optimizer' is not defined`，因为相关函数未导入。
* 2025-05-26 16:41:27 - 发现 `run_quick_test_evaluation.py` 存在 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`，原因是 `Config` 类中的模型配置未在 `LOOKBACK_WINDOW` 和 `PREDICTION_HORIZON` 动态更新后重新计算。
* 2025-05-26 16:43:29 - 发现 `run_quick_test_evaluation.py` 存在 `TypeError: evaluate_model() missing 2 required positional arguments: 'device' and 'scaler'`，与 `evaluate_model` 函数的调用有关。
* 2025-05-26 16:45:28 - 发现 `run_quick_test_evaluation.py` 存在 `AttributeError: 'numpy.ndarray' object has no attribute 'eval'`，原因是 `evaluate_model` 函数被错误地传入了 NumPy 数组而不是模型对象。
* 2025-05-26 16:49:04 - 发现 `run_quick_test_evaluation.py` 存在 `AttributeError: 'numpy.ndarray' object has no attribute 'eval'`，原因是 `evaluate_model` 函数被错误地传入了 NumPy 数组而不是模型对象。
* 2025-05-26 16:51:48 - 再次发现 `run_quick_test_evaluation.py` 存在 `AttributeError: 'numpy.ndarray' object has no attribute 'eval'`，`evaluate_model` 函数仍被错误地传入 NumPy 数组。
* 2025-05-26 16:59:15 - 在 `src/evaluator.py` 的 `predict` 函数中添加了类型检查，以确保 `model` 参数是 `torch.nn.Module` 的实例。这有助于在更早的阶段捕获 `AttributeError`。
* 2025-05-26 16:59:15 - **重要发现**: `AttributeError: 'numpy.ndarray' object has no attribute 'eval'` 的根本原因在于 `src/trainer.py` 中的 `_validate_epoch` 函数错误地将 `preds_all` (一个 NumPy 数组) 传递给了 `evaluate_model` 函数的 `model` 参数。需要进一步修复 `src/trainer.py`。
* 2025-05-26 17:00:28 - 在 `src/evaluator.py` 的 `predict` 函数中添加了类型检查，以确保 `model` 参数是 `torch.nn.Module` 的实例。
* 2025-05-26 17:01:33 - 修复了 `src/trainer.py` 中 `_validate_epoch` 函数对 `evaluate_model` 的错误调用，现在传递正确的 `model` 和 `dataloader`。
* 2025-05-26 17:04:31 - 修复 `src/evaluator.py` 中的 `calculate_metrics` 函数，使其接受 `metrics_list` 参数。
* 2025-05-26 17:04:31 - 修复 `src/evaluator.py` 中的 `evaluate_model` 函数，使其接受 `config_obj` 参数，并将 `config_obj.METRICS` 和 `config_obj.SIMILARITY_METRIC` 传递给相应的函数。
* 2025-05-26 17:04:31 - 修复 `src/trainer.py` 中的 `BaseTrainer.__init__` 方法，使其接受 `config_obj` 参数并将其存储为 `self.config`。
* 2025-05-26 17:04:31 - 修复 `src/trainer.py` 中的 `StandardTrainer.__init__` 方法，使其接受 `config_obj` 参数并将其传递给 `super().__init__`。
* 2025-05-26 17:04:31 - 修复 `src/trainer.py` 中的 `StandardTrainer._validate_epoch` 方法，在调用 `evaluate_model` 时传递 `self.config`。
* 2025-05-26 17:05:24 - 修复 `run_quick_test_evaluation.py` 中的 `run_experiment` 函数，在实例化 `StandardTrainer` 时传递 `config_obj`。
* 2025-05-26 17:07:48 - 修复 `run_quick_test_evaluation.py` 中所有对 `evaluate_model` 的调用，传递 `config` 对象。
* 2025-05-26 17:08:49 - 修复 `src/trainer.py` 中的 `train` 函数，正确解包 `_validate_epoch` 返回的元组。
* 2025-05-26 17:10:06 - 修复 `src/trainer.py` 中的 `_validate_epoch` 函数，使其返回 4 个独立的元素，与 `train` 函数的解包方式匹配。
* 2025-05-26 17:11:01 - 修复 `run_quick_test_evaluation.py` 中所有对 `evaluate_model` 的调用，移除 `logger` 参数。
* 2025-05-26 17:11:56 - 修复 `run_quick_test_evaluation.py` 中的 `run_experiment` 函数，添加创建 `results` 和 `plots` 目录的代码。
* 2025-05-26 17:13:05 - 修复 `run_quick_test_evaluation.py` 中的 `run_experiment` 函数，在每次稳定性运行迭代中初始化 `results` 和 `similarity_results` 字典。
* 2025-05-26 17:16:36 - 修复 `src/trainer.py` 中的 `RDT_Trainer._validate_epoch` 方法，在调用 `evaluate_model` 时传递所有必需的参数。
* 2025-05-26 17:18:40 - 修复 `run_quick_test_evaluation.py` 中所有对 `RDT_Trainer` 的实例化，传递 `config` 对象。
* 2025-05-26 17:20:11 - 修复 `run_quick_test_evaluation.py` 中 `TaskOnly` 模型的 `RDT_Trainer` 实例化，传递 `config` 对象。
* 2025-05-26 17:21:08 - 修复 `src/trainer.py` 中的 `RDT_Trainer._validate_epoch` 方法，将 `calculate_similarity_metrics` 的 `metric` 参数名修正为 `metric_type`。
* 2025-05-26 17:23:05 - 修复 `src/trainer.py` 中的 `RDT_Trainer._validate_epoch` 函数，使其返回 4 个独立的元素，与 `train` 函数的解包方式匹配。
* 2025-05-26 17:24:38 - 修复 `src/trainer.py` 中的 `RDT_Trainer._validate_epoch` 方法，将相似度指标直接合并到 `val_metrics` 字典中。
* 2025-05-26 17:29:42 - 修复 `run_evaluation_experiments.py` 中的 `run_experiment` 函数，在每次稳定性运行迭代中初始化 `results` 和 `similarity_results` 字典。
* 2025-05-26 17:29:42 - 修复 `run_evaluation_experiments.py` 中的 `run_experiment` 函数，添加创建 `results` 和 `plots` 目录的代码。
* 2025-05-26 17:29:42 - 修复 `run_evaluation_experiments.py` 中 `StandardTrainer` 和 `RDT_Trainer` 的实例化，传递 `config_obj`。
* 2025-05-26 17:29:42 - 修复 `run_evaluation_experiments.py` 中所有对 `evaluate_model` 的调用，传递 `config` 对象并移除 `logger` 参数。
* 2025-05-26 17:40:00 - 完成了核心训练指标（训练损失、验证损失、学习率、梯度范数和评估指标）的可视化功能。
* 2025-05-26 17:41:00 - 完成了模型权重和偏置分布的可视化功能。
* 2025-05-26 17:43:00 - 完成了增强预测结果与真实值对比的可视化功能（包括残差分析和误差分布图）。
* 2025-05-26 17:45:00 - 完成了详细的参数记录机制，包括实验元数据、模型架构参数、训练配置参数和数据处理参数的保存。
* [2025-05-26 18:07:13] - 修复 `src/utils.py` 中 `plot_predictions` 函数的 `title` 和 `save_path` 未定义警告，通过修改函数签名并正确传递参数。
* 2025-05-26 21:27:19 - 修改 `run_evaluation_experiments.py` 以实现噪音注入和去噪平滑的配置调整。
* 2025-05-27 00:14:58 - 完成 `README.md` 文档更新任务。
* [2025-05-27 01:21:37] - 修改 `run_evaluation_experiments.py` 脚本，使其在保存模型和结果时，使用新的命名格式：`results/数据集_Teacher模型_Student模型_h预测窗口长度_noise噪音水平_smooth平滑系数_训练时间戳`。
* [2025-05-27 02:05:00] - 优化了 `run_evaluation_experiments.py` 文件的结果保存结构，现在采用更规范的多层级目录结构：
  1. 根目录使用时间戳命名 `results/experiments_YYYYMMDD_HHMMSS/`
  2. 实验总览文件 `experiment_overview.json` 和结果CSV文件保存在根目录
  3. 实验组合目录不再包含时间戳，格式为 `dataset_Teacher_Student_hX_noiseN_smoothM`
  4. 稳定性运行子目录格式标准化为 `runX_seed_YY`
  5. 日志文件保存在 `log/experiment_log_YYYYMMDD_HHMMSS.log`