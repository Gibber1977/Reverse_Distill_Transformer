import os
import json
import pandas as pd
import torch
import numpy as np
from datetime import datetime

from src.config import Config
from src.data_handler import TimeSeriesDataset, load_and_preprocess_data
from src.models import get_model
from src.trainer import StandardTrainer, RDT_Trainer, get_optimizer, get_loss_function
from src.schedulers import get_alpha_scheduler
from src.evaluator import evaluate_model, calculate_similarity_metrics
from src.utils import set_seed, setup_logging, save_results_to_csv, save_experiment_metadata # Removed save_plot

# --- 快速测试实验配置 ---
DATASETS = {
    'exchange_rate': './data/exchange_rate.csv',
    # 'national_illness': './data/national_illness.csv',
    'weather': './data/weather.csv',
    # 'ETTh1': './data/ETT-small/ETTh1.csv',
    # 'ETTh2': './data/ETT-small/ETTh2.csv',
    # 'ETTm1': './data/ETT-small/ETTm1.csv',
    # 'ETTm2': './data/ETT-small/ETTm2.csv',
    'PEMS_0':'./data/PEMS_0.csv'
}
PREDICTION_HORIZONS = [336,720]
LOOKBACK_WINDOW = 192
EPOCHS = 50 # 减少epochs以加快测试
STABILITY_RUNS = 3 # 减少运行次数

# 模型组合: (Teacher, Student)
MODELS = [
    ('DLinear', 'PatchTST'),     # 测试 PatchTST 作为独立学生模型

]

# 噪音注入评估配置 (减少噪音水平)
NOISE_LEVELS = [0.01,0.02,0.05,0.10,0.15,0.20]
NOISE_TYPE = 'gaussian'

# 去噪平滑评估配置 (减少平滑系数)
SMOOTHING_FACTORS = [0.01,0.02,0.05,0.10,0.15,0.20]
SMOOTHING_METHOD = 'moving_average'

# --- 主实验函数 (与 run_evaluation_experiments.py 相同，但使用快速测试配置) ---
def run_experiment(
    dataset_name,
    dataset_path,
    pred_horizon,
    lookback_window,
    epochs,
    stability_runs,
    teacher_model_name,
    student_model_name,
    noise_level=0,
    noise_type=None,
    smoothing_factor=0,
    smoothing_method=None,
    experiment_type='standard', # 'standard', 'noise_injection', 'denoising_smoothing'
    logger=None
):
    logger.info(f"--- Running Quick Test Experiment: {experiment_type} ---")
    logger.info(f"Dataset: {dataset_name}, Prediction Horizon: {pred_horizon}, "
                f"Teacher: {teacher_model_name}, Student: {student_model_name}")
    if noise_type:
        logger.info(f"Noise: {noise_type} at level {noise_level}")
    if smoothing_method:
        logger.info(f"Smoothing: {smoothing_method} with factor {smoothing_factor}")

    all_run_results = []
    all_similarity_results = []

    for run_idx in range(stability_runs):
        logger.info(f"--- Stability Run {run_idx + 1}/{stability_runs} ---")
        set_seed(42 + run_idx) # 每次运行使用不同的种子
        results = {} # 初始化当前运行的结果字典
        similarity_results = {} # 初始化当前运行的相似度结果字典

        # 加载和预处理数据
        config = Config()
        config.LOOKBACK_WINDOW = lookback_window
        config.PREDICTION_HORIZON = pred_horizon
        config.TEACHER_MODEL_NAME = teacher_model_name
        config.STUDENT_MODEL_NAME = student_model_name
        config.EPOCHS = epochs
        config.TRAIN_NOISE_INJECTION_LEVEL = noise_level
        config.NOISE_TYPE = noise_type
        config.SMOOTHING_FACTOR = smoothing_factor
        config.SMOOTHING_METHOD = smoothing_method
        config.SIMILARITY_METRIC = 'cosine_similarity' # 固定为余弦相似度

        # 设置 EXOGENOUS_COLS
        if dataset_name == 'exchange_rate': # 示例：只为 exchange_rate 添加协变量
            config.EXOGENOUS_COLS = ['0', '1', '2', '3', '4', '5', '6']
            logger.info(f"Setting EXOGENOUS_COLS for {dataset_name}: {config.EXOGENOUS_COLS}")
        else:
            config.EXOGENOUS_COLS = [] # 其他数据集不使用
            logger.info(f"Setting EXOGENOUS_COLS for {dataset_name}: {config.EXOGENOUS_COLS}")

        # 更新模型配置，确保 LOOKBACK_WINDOW 和 PREDICTION_HORIZON 的变化生效
        config.update_model_configs()

        # 确保结果目录存在
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        # Removed: os.makedirs(os.path.join(config.RESULTS_DIR, "plots"), exist_ok=True)

        # 根据实验类型调整数据处理
        if experiment_type == 'noise_injection':
            pass
        elif experiment_type == 'denoising_smoothing':
            pass
        else: # standard experiment, ensure no noise or smoothing
            config.TRAIN_NOISE_INJECTION_LEVEL = 0
            config.NOISE_TYPE = None
            config.SMOOTHING_FACTOR = 0
            config.SMOOTHING_METHOD = None

        # 从 dataset_path 或 dataset_name 中提取文件名
        dataset_filename = os.path.basename(dataset_path)
        # 从 DATASET_TIME_FREQ_MAP 中查找时间频率
        time_freq_for_dataset = config.DATASET_TIME_FREQ_MAP.get(dataset_filename, 'h') # 默认为 'h'
        config.TIME_FREQ = time_freq_for_dataset # 更新 config 中的 TIME_FREQ (虽然 data_handler 现在用传入的)
        logger.info(f"Using time frequency '{time_freq_for_dataset}' for dataset '{dataset_filename}'")

        train_loader, val_loader, test_loader, scaler, n_features = load_and_preprocess_data(
            dataset_path, config, logger, time_freq=time_freq_for_dataset
        )
        config.N_FEATURES = n_features
        config.update_model_configs() # 确保在设置 n_features 后调用此函数

        # --- 设备设置 ---
        actual_device_str = config.DEVICE
        if "cuda" in actual_device_str.lower() and not torch.cuda.is_available():
            logger.warning(f"配置请求 CUDA ({actual_device_str}) 但 CUDA 不可用。将使用 CPU。")
            actual_device_str = "cpu"
        elif "cuda" in actual_device_str.lower() and torch.cuda.is_available():
            logger.info(f"CUDA 可用，将尝试使用设备: {actual_device_str}")
        else: # CPU or other
            actual_device_str = "cpu"
            logger.info(f"将使用 CPU。")

        device = torch.device(actual_device_str)
        config.DEVICE = str(device) # 更新配置中的实际使用设备
        logger.info(f"实验最终将在设备上运行: {device}")
        # --- 设备设置结束 ---

        # 获取学生模型
        student_model_instance = get_model(student_model_name, config).to(device) # Renamed to avoid conflict

        teacher_model = None
        teacher_metrics = {}
        teacher_preds_original = None

        if teacher_model_name is not None:
            teacher_model = get_model(teacher_model_name, config).to(device)
            # 训练 Teacher 模型
            logger.info(f"Training Teacher Model: {teacher_model_name}")
            teacher_trainer = StandardTrainer(
                model=teacher_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=get_optimizer(teacher_model, config),
                loss_fn=get_loss_function(config),
                device=config.DEVICE,
                epochs=config.EPOCHS,
                patience=config.PATIENCE,
                model_save_path=os.path.join(config.RESULTS_DIR, f"{teacher_model_name}_teacher_model.pt"),
                scaler=scaler,
                config_obj=config,
                model_name=teacher_model_name
            )
            teacher_trainer.train()

            # 评估 Teacher 模型
            teacher_metrics, teacher_true_original, teacher_preds_original = evaluate_model(
                teacher_model, test_loader, config.DEVICE, scaler, config, logger,
                model_name=teacher_model_name # Removed plots_dir
            )
            for metric, value in teacher_metrics.items():
                results[f'Teacher_{metric}'] = value
        else:
            logger.info("Teacher model name is None, skipping teacher model training and evaluation.")
            # teacher_model is already None
            teacher_metrics = {} # Ensure it's an empty dict
            teacher_preds_original = None # Ensure it's None

        # 训练 TaskOnly 模型
        # Re-initialize student model for TaskOnly to ensure fresh state if it was used by RDT/Follower before
        current_student_model_task_only = get_model(student_model_name, config).to(device)
        if teacher_model is None:
            logger.info(f"Training TaskOnly Model (Student: {student_model_name}) as a standard student model.")
            task_only_trainer = StandardTrainer(
                model=current_student_model_task_only,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=get_optimizer(current_student_model_task_only, config),
                loss_fn=get_loss_function(config),
                device=config.DEVICE,
                epochs=config.EPOCHS,
                patience=config.PATIENCE,
                model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_task_only_standalone_model.pt"),
                scaler=scaler,
                config_obj=config,
                model_name=f"{student_model_name}_TaskOnly_Standalone"
            )
            task_only_trainer.train()
            # 评估 TaskOnly (Standalone) 模型
            task_only_metrics, _, task_only_preds_original = evaluate_model(
                task_only_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
                model_name=f"{student_model_name}_TaskOnly_Standalone",
                teacher_predictions_original=None # No teacher for comparison
            )
        else:
            logger.info(f"Training TaskOnly Model (Student: {student_model_name}, alpha=1, with Teacher: {teacher_model_name})")
            config.ALPHA_SCHEDULE = 'constant'
            config.CONSTANT_ALPHA = 1.0
            task_only_trainer = RDT_Trainer(
                teacher_model=teacher_model, # This will be the trained teacher model
                student_model=current_student_model_task_only,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=get_optimizer(current_student_model_task_only, config),
                task_loss_fn=get_loss_function(config),
                distill_loss_fn=get_loss_function(config), # Not used when alpha=1 but required
                alpha_scheduler=get_alpha_scheduler(config),
                device=config.DEVICE,
                epochs=config.EPOCHS,
                patience=config.PATIENCE,
                model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_task_only_model.pt"),
                scaler=scaler,
                config_obj=config,
                model_name=f"{student_model_name}_TaskOnly"
            )
            task_only_trainer.train()
            # 评估 TaskOnly 模型
            task_only_metrics, _, task_only_preds_original = evaluate_model(
                task_only_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
                model_name=f"{student_model_name}_TaskOnly",
                teacher_predictions_original=teacher_preds_original, # Compare with actual teacher
            )

        for metric, value in task_only_metrics.items():
            results[f'TaskOnly_{metric}'] = value


        if teacher_model is not None:
            # Re-initialize student model for Follower
            current_student_model_follower = get_model(student_model_name, config).to(device)
            # 训练 Follower 模型 (alpha=0)
            logger.info(f"Training Follower Model (Student: {student_model_name}, alpha=0, with Teacher: {teacher_model_name})")
            config.ALPHA_SCHEDULE = 'constant'
            config.CONSTANT_ALPHA = 0.0
            follower_trainer = RDT_Trainer(
                teacher_model=teacher_model,
                student_model=current_student_model_follower,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=get_optimizer(current_student_model_follower, config),
                task_loss_fn=get_loss_function(config), # Not used when alpha=0 but required
                distill_loss_fn=get_loss_function(config),
                alpha_scheduler=get_alpha_scheduler(config),
                device=config.DEVICE,
                epochs=config.EPOCHS,
                patience=config.PATIENCE,
                model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_follower_model.pt"),
                scaler=scaler,
                config_obj=config,
                model_name=f"{student_model_name}_Follower"
            )
            follower_trainer.train()

            # 评估 Follower 模型
            follower_metrics, _, follower_preds_original = evaluate_model(
                follower_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
                model_name=f"{student_model_name}_Follower",
                teacher_predictions_original=teacher_preds_original,
            )
            for metric, value in follower_metrics.items():
                results[f'Follower_{metric}'] = value

            # Re-initialize student model for RDT
            current_student_model_rdt = get_model(student_model_name, config).to(device)
            # 训练 RDT 模型 (alpha 动态调度)
            logger.info(f"Training RDT Model (Student: {student_model_name}, dynamic alpha, with Teacher: {teacher_model_name})")
            config.ALPHA_SCHEDULE = 'linear' # Or other dynamic schedule
            config.CONSTANT_ALPHA = None # Ensure it's not overriding
            rdt_trainer = RDT_Trainer(
                teacher_model=teacher_model,
                student_model=current_student_model_rdt,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=get_optimizer(current_student_model_rdt, config),
                task_loss_fn=get_loss_function(config),
                distill_loss_fn=get_loss_function(config),
                alpha_scheduler=get_alpha_scheduler(config),
                device=config.DEVICE,
                epochs=config.EPOCHS,
                patience=config.PATIENCE,
                model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_rdt_model.pt"),
                scaler=scaler,
                config_obj=config,
                model_name=f"{student_model_name}_RDT"
            )
            rdt_trainer.train()

            # 评估 RDT 模型
            rdt_metrics, _, rdt_preds_original = evaluate_model(
                rdt_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
                model_name=f"{student_model_name}_RDT",
                teacher_predictions_original=teacher_preds_original,
            )
            for metric, value in rdt_metrics.items():
                results[f'RDT_{metric}'] = value
        else:
            logger.info("Teacher model is None, skipping Follower and RDT model training and evaluation.")
            # Ensure metrics for Follower and RDT are not added or are NaN
            # For simplicity, we are not adding them if skipped.
            # If consistency in CSV columns is needed, add NaN values here.
            # e.g., results['Follower_mse'] = np.nan
            # results['RDT_mse'] = np.nan
            # ... and so on for all expected metrics
            follower_metrics = {} # Ensure empty if skipped
            rdt_metrics = {}      # Ensure empty if skipped


        # 从指标字典中提取相似度结果
        # teacher_metrics will be empty if teacher_model is None
        for key, value in teacher_metrics.items(): # teacher_metrics is already defined
            if 'similarity' in key:
                similarity_results[f'Teacher_{key}'] = value
        
        # task_only_metrics is defined regardless of teacher_model
        for key, value in task_only_metrics.items():
            if 'similarity' in key: # This will only populate if teacher_model was present for TaskOnly RDT training
                similarity_results[f'TaskOnly_{key}'] = value
        
        if teacher_model is not None: # follower_metrics and rdt_metrics only exist if teacher_model was present
            for key, value in follower_metrics.items():
                if 'similarity' in key:
                    similarity_results[f'Follower_{key}'] = value
            for key, value in rdt_metrics.items():
                if 'similarity' in key:
                    similarity_results[f'RDT_{key}'] = value
        else: # If no teacher, similarity for follower and RDT is not applicable
            # Optionally add NaN for consistency if needed for CSV columns
            # similarity_results['Follower_cosine_similarity'] = np.nan
            # similarity_results['RDT_cosine_similarity'] = np.nan
            pass # No similarity to record for these if no teacher

        # 记录当前运行的元数据
        run_metadata = {
            'dataset': dataset_name,
            'pred_horizon': pred_horizon,
            'lookback_window': lookback_window,
            'epochs': epochs,
            'teacher_model': teacher_model_name,
            'student_model': student_model_name,
            'noise_level': noise_level,
            'noise_type': noise_type,
            'smoothing_factor': smoothing_factor,
            'smoothing_method': smoothing_method,
            'experiment_type': experiment_type,
            'run_idx': run_idx + 1
        }
        all_run_results.append({**run_metadata, **results})
        all_similarity_results.append({**run_metadata, **similarity_results})

    return all_run_results, all_similarity_results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_quick_test_no_plots/experiments_{timestamp}" # 快速测试结果保存到单独目录
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join("log", f"quick_test_no_plots_experiment_log_{timestamp}.log")
    logger = setup_logging(log_file)

    # 保存实验元数据
    current_config = Config()
    current_config.LOOKBACK_WINDOW = LOOKBACK_WINDOW
    current_config.PREDICTION_HORIZON = PREDICTION_HORIZONS[0]
    current_config.TEACHER_MODEL_NAME = MODELS[0][0]
    current_config.STUDENT_MODEL_NAME = MODELS[0][1]
    current_config.EPOCHS = EPOCHS
    current_config.STABILITY_RUNS = STABILITY_RUNS
    current_config.ROBUSTNESS_NOISE_LEVELS = NOISE_LEVELS
    current_config.SMOOTHING_FACTORS = SMOOTHING_FACTORS
    current_config.update_model_configs()
    
    save_experiment_metadata(current_config, results_dir, f"quick_test_no_plots_experiment_{timestamp}")

    all_experiment_results = []
    all_experiment_similarity_results = []

    logger.info("Starting quick evaluation experiments (no plots)...")

    for dataset_name, dataset_path in DATASETS.items():
        for pred_horizon in PREDICTION_HORIZONS:
            for teacher_model, student_model in MODELS:
                logger.info(f"\n--- Running for Dataset: {dataset_name}, Horizon: {pred_horizon}, "
                            f"Models: {teacher_model}/{student_model} ---")

                # 1. 标准评估 (无噪音, 无平滑)
                logger.info("\n--- Running Standard Evaluation ---")
                run_results, sim_results = run_experiment(
                    dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                    teacher_model, student_model,
                    experiment_type='standard', logger=logger
                )
                all_experiment_results.extend(run_results)
                all_experiment_similarity_results.extend(sim_results)

                # 2. 噪音注入评估
                logger.info("\n--- Running Noise Injection Evaluation ---")
                for noise_level in NOISE_LEVELS:
                    run_results, sim_results = run_experiment(
                        dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                        teacher_model, student_model,
                        noise_level=noise_level, noise_type=NOISE_TYPE,
                        experiment_type='noise_injection', logger=logger
                    )
                    all_experiment_results.extend(run_results)
                    all_experiment_similarity_results.extend(sim_results)

                # 3. 去噪平滑评估
                logger.info("\n--- Running Denoising Smoothing Evaluation ---")
                for smoothing_factor in SMOOTHING_FACTORS:
                    run_results, sim_results = run_experiment(
                        dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                        teacher_model, student_model,
                        smoothing_factor=smoothing_factor, smoothing_method=SMOOTHING_METHOD,
                        experiment_type='denoising_smoothing', logger=logger
                    )
                    all_experiment_results.extend(run_results)
                    all_experiment_similarity_results.extend(sim_results)

    # 保存所有结果
    results_df = pd.DataFrame(all_experiment_results)
    similarity_df = pd.DataFrame(all_experiment_similarity_results)

    results_csv_path = os.path.join(results_dir, "quick_test_no_plots_experiment_results.csv")
    similarity_csv_path = os.path.join(results_dir, "quick_test_no_plots_similarity_results.csv")

    save_results_to_csv(results_df, results_csv_path, logger)
    save_results_to_csv(similarity_df, similarity_csv_path, logger)

    logger.info(f"All quick test experiment results saved to: {results_csv_path}")
    logger.info(f"All quick test similarity results saved to: {similarity_csv_path}")

    # Removed visualization generation
    # logger.info("\n--- Generating Quick Test Visualizations ---")
    # plot_noise_evaluation(results_df, similarity_df, results_dir, logger)
    # plot_smoothing_evaluation(results_df, similarity_df, results_dir, logger)
    # logger.info("All quick test visualizations generated.")
    
    logger.info("Quick evaluation experiments (no plots) completed.")

# Removed plot_noise_evaluation function
# Removed plot_smoothing_evaluation function

if __name__ == "__main__":
    main()