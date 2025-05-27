import os
import json
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

from src.config import Config
from src.data_handler import TimeSeriesDataset, load_and_preprocess_data
from src.models import get_model
from src.trainer import StandardTrainer, RDT_Trainer, get_optimizer, get_loss_function
from src.schedulers import get_alpha_scheduler
from src.evaluator import evaluate_model, calculate_similarity_metrics
from src.utils import set_seed, setup_logging, save_results_to_csv, save_plot, save_experiment_metadata

# --- 实验配置 ---
DATASETS = {
    'exchange_rate': './data/exchange_rate.csv',
    # 'national_illness': './data/national_illness.csv',
    # 'weather': './data/weather.csv',
    'ETTh1': './data/ETT-small/ETTh1.csv',
    'ETTh2': './data/ETT-small/ETTh2.csv',
    # 'ETTm1': './data/ETT-small/ETTm1.csv',
    # 'ETTm2': './data/ETT-small/ETTm2.csv',
}

# PREDICTION_HORIZONS = [24, 96, 192, 336, 720]
PREDICTION_HORIZONS = [96, 336, 720]
LOOKBACK_WINDOW = 192
EPOCHS = 100
STABILITY_RUNS = 5

# 模型组合: (Teacher, Student)
MODELS = [('DLinear', 'PatchTST')]

# 噪音注入评估配置
# NOISE_LEVELS = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
NOISE_LEVELS = [ 0.02, 0.10, 0.20]
NOISE_TYPE = 'gaussian'

# 去噪平滑评估配置
SMOOTHING_FACTORS = [ 0.10,  0.30, 0.5]
SMOOTHING_METHOD = 'moving_average'
# --- 主实验函数 ---
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
    smoothing_weight_smoothing=0.0,
    smoothing_method=None,
    experiment_type='standard', # 'standard', 'noise_injection', 'denoising_smoothing'
    logger=None,
    base_output_dir=None, # 新增参数
    pbar=None # 新增 pbar 参数
):
    logger.info(f"--- Running Experiment: {experiment_type} ---")
    logger.info(f"Dataset: {dataset_name}, Prediction Horizon: {pred_horizon}, "
                f"Teacher: {teacher_model_name}, Student: {student_model_name}")
    if noise_type:
        logger.info(f"Noise: {noise_type} at level {noise_level}")
    if smoothing_method:
        logger.info(f"Smoothing: {smoothing_method} with weight_smoothing {smoothing_weight_smoothing}")

    all_run_results = []
    all_similarity_results = []

    for run_idx in range(stability_runs):
        current_seed = 42 + run_idx
        
        # 生成唯一的实验结果子文件夹
        teacher_name_for_dir = teacher_model_name if teacher_model_name else "NoTeacher"
        base_experiment_name = (
            f"{dataset_name}_{teacher_name_for_dir}_{student_model_name}_"
            f"h{pred_horizon}_noise{noise_level}_smooth_w{smoothing_weight_smoothing}"
        )
        
        if base_output_dir is None:
            base_output_dir = "results"

        parent_experiment_dir = os.path.join(base_output_dir, base_experiment_name)
        os.makedirs(parent_experiment_dir, exist_ok=True)

        experiment_results_dir = os.path.join(parent_experiment_dir, f"run{run_idx}_seed_{current_seed}")
        
        # 断点续训逻辑
        completion_marker_file = os.path.join(experiment_results_dir, "run_completed.txt")
        if os.path.exists(completion_marker_file):
            logger.info(f"--- Stability Run {run_idx + 1}/{stability_runs} (Seed: {current_seed}) already completed. Skipping. ---")
            if pbar:
                pbar.update(1)
            continue # 跳过已完成的运行

        logger.info(f"--- Stability Run {run_idx + 1}/{stability_runs} (Seed: {current_seed}) ---")
        set_seed(current_seed)
        results = {}
        similarity_results = {}

        # 加载和预处理数据
        config = Config()
        config.LOOKBACK_WINDOW = lookback_window
        config.PREDICTION_HORIZON = pred_horizon
        config.TEACHER_MODEL_NAME = teacher_model_name
        config.STUDENT_MODEL_NAME = student_model_name
        config.EPOCHS = epochs
        config.TRAIN_NOISE_INJECTION_LEVEL = noise_level
        config.NOISE_TYPE = noise_type
        config.SMOOTHING_FACTOR = 24 # 为 moving_average 设置一个默认窗口大小
        config.SMOOTHING_WEIGHT_SMOOTHING = smoothing_weight_smoothing
        config.SMOOTHING_METHOD = smoothing_method
        config.SIMILARITY_METRIC = 'cosine_similarity' # 保持与run_quick_test_evaluation.py一致
        config.RESULTS_DIR = experiment_results_dir # 设置为新的子文件夹
        config.update_model_configs() # 更新依赖于这些值的模型配置

        # 确保结果和绘图目录存在
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, "plots"), exist_ok=True)

        # 根据实验类型调整数据处理
        if experiment_type == 'noise_injection':
            # 噪音注入在 load_and_preprocess_data 内部处理
            config.VAL_NOISE_INJECTION_LEVEL = noise_level
        elif experiment_type == 'denoising_smoothing':
            # 平滑处理在 load_and_preprocess_data 内部处理
            pass
        else: # standard experiment, ensure no noise or smoothing
            config.TRAIN_NOISE_INJECTION_LEVEL = 0
            config.NOISE_TYPE = None
            config.SMOOTHING_FACTOR = 0
            config.SMOOTHING_METHOD = None
            config.SMOOTHING_APPLY_TEST = False # Standard experiment should not smooth test data
        
        if experiment_type == 'denoising_smoothing':
            config.SMOOTHING_APPLY_TEST = True # Denoising smoothing should apply smoothing to test data
            config.SMOOTHING_APPLY_TRAIN = True
            config.SMOOTHING_APPLY_VAL = True
        else:
            config.SMOOTHING_APPLY_TEST = False # Other experiments should not smooth test data
            config.SMOOTHING_APPLY_TRAIN = False
            config.SMOOTHING_APPLY_VAL = False

        train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(
            dataset_path, config, logger
        )

        # 获取模型
        teacher_model = get_model(teacher_model_name, config).to(config.DEVICE)
        student_model = get_model(student_model_name, config).to(config.DEVICE)

        # 训练 Teacher 模型 (如果需要)
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
            config_obj=config, # Pass config object
            model_name=teacher_model_name
        )
        teacher_trainer.train()
        # teacher_preds_on_test, teacher_true_on_test = teacher_trainer.predict(test_loader) # Removed

        # 评估 Teacher 模型
        teacher_metrics, teacher_true_original, teacher_preds_original = evaluate_model(
            teacher_model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=teacher_model_name, plots_dir=os.path.join(config.RESULTS_DIR, "plots"),
            dataset_type="Test Set"
        )
        for metric, value in teacher_metrics.items():
            results[f'Teacher_{metric}'] = value

        # 训练 TaskOnly 模型 (alpha=1)
        logger.info(f"Training TaskOnly Model (Student: {student_model_name}, alpha=1)")
        config.ALPHA_SCHEDULE = 'constant' # Use class attribute
        config.CONSTANT_ALPHA = 1.0 # Use class attribute
        task_only_trainer = RDT_Trainer(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=get_optimizer(student_model, config),
            task_loss_fn=get_loss_function(config),
            distill_loss_fn=get_loss_function(config),
            alpha_scheduler=get_alpha_scheduler(config),
            device=config.DEVICE,
            epochs=config.EPOCHS,
            patience=config.PATIENCE,
            model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_task_only_model.pt"),
            scaler=scaler,
            config_obj=config, # Pass config object
            model_name=f"{student_model_name}_TaskOnly"
        )
        task_only_trainer.train()
        # task_only_preds, task_only_true = task_only_trainer.predict(test_loader) # Removed

        # 评估 TaskOnly 模型
        task_only_metrics, _, task_only_preds_original = evaluate_model(
            task_only_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=f"{student_model_name}_TaskOnly", plots_dir=os.path.join(config.RESULTS_DIR, "plots"),
            teacher_predictions_original=teacher_preds_original, # Pass teacher's original predictions
            dataset_type="Test Set"
        )
        for metric, value in task_only_metrics.items():
            results[f'TaskOnly_{metric}'] = value
        # 相似度已包含在 task_only_metrics 中

        # 训练 Follower 模型 (alpha=0)
        logger.info(f"Training Follower Model (Student: {student_model_name}, alpha=0)")
        config.ALPHA_SCHEDULE = 'constant' # Use class attribute
        config.CONSTANT_ALPHA = 0.0 # Use class attribute
        follower_trainer = RDT_Trainer(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=get_optimizer(student_model, config),
            task_loss_fn=get_loss_function(config),
            distill_loss_fn=get_loss_function(config),
            alpha_scheduler=get_alpha_scheduler(config),
            device=config.DEVICE,
            epochs=config.EPOCHS,
            patience=config.PATIENCE,
            model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_follower_model.pt"),
            scaler=scaler,
            config_obj=config, # Pass config object
            model_name=f"{student_model_name}_Follower"
        )
        follower_trainer.train()
        # follower_preds, follower_true = follower_trainer.predict(test_loader) # Removed

        # 评估 Follower 模型
        follower_metrics, _, follower_preds_original = evaluate_model(
            follower_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=f"{student_model_name}_Follower", plots_dir=os.path.join(config.RESULTS_DIR, "plots"),
            teacher_predictions_original=teacher_preds_original, # Pass teacher's original predictions
            dataset_type="Test Set"
        )
        for metric, value in follower_metrics.items():
            results[f'Follower_{metric}'] = value
        # 相似度已包含在 follower_metrics 中

        # 训练 RDT 模型 (alpha 动态调度)
        logger.info(f"Training RDT Model (Student: {student_model_name}, dynamic alpha)")
        config.ALPHA_SCHEDULE = 'linear' # 示例使用线性调度
        config.CONSTANT_ALPHA = None # 确保不是固定alpha
        rdt_trainer = RDT_Trainer(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=get_optimizer(student_model, config),
            task_loss_fn=get_loss_function(config),
            distill_loss_fn=get_loss_function(config),
            alpha_scheduler=get_alpha_scheduler(config),
            device=config.DEVICE,
            epochs=config.EPOCHS,
            patience=config.PATIENCE,
            model_save_path=os.path.join(config.RESULTS_DIR, f"{student_model_name}_rdt_model.pt"),
            scaler=scaler,
            config_obj=config, # Pass config object
            model_name=f"{student_model_name}_RDT"
        )
        rdt_trainer.train()
        # rdt_preds, rdt_true = rdt_trainer.predict(test_loader) # Removed

        # 评估 RDT 模型
        rdt_metrics, _, rdt_preds_original = evaluate_model(
            rdt_trainer.model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=f"{student_model_name}_RDT", plots_dir=os.path.join(config.RESULTS_DIR, "plots"),
            teacher_predictions_original=teacher_preds_original, # Pass teacher's original predictions
            dataset_type="Test Set"
        )
        for metric, value in rdt_metrics.items():
            results[f'RDT_{metric}'] = value
        # 相似度已包含在 rdt_metrics 中

        # 从指标字典中提取相似度结果
        for key, value in teacher_metrics.items():
            if 'similarity' in key:
                similarity_results[f'Teacher_{key}'] = value
        for key, value in task_only_metrics.items():
            if 'similarity' in key:
                similarity_results[f'TaskOnly_{key}'] = value
        for key, value in follower_metrics.items():
            if 'similarity' in key:
                similarity_results[f'Follower_{key}'] = value
        for key, value in rdt_metrics.items():
            if 'similarity' in key:
                similarity_results[f'RDT_{key}'] = value

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
            'smoothing_weight_smoothing': smoothing_weight_smoothing,
            'smoothing_method': smoothing_method,
            'experiment_type': experiment_type,
            'run_idx': run_idx + 1
        }
        all_run_results.append({**run_metadata, **results})
        all_similarity_results.append({**run_metadata, **similarity_results})

        # 标记当前运行已完成
        with open(completion_marker_file, 'w') as f:
            f.write("completed")
        logger.info(f"--- Stability Run {run_idx + 1}/{stability_runs} (Seed: {current_seed}) completed and marked. ---")
        
        if pbar:
            pbar.update(1)

    return all_run_results, all_similarity_results

def main():
    # 创建带时间戳的根目录
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)

    # 确保log文件夹存在
    os.makedirs("log", exist_ok=True)
    log_file = os.path.join("log", "experiment_log.log")
    logger = setup_logging(log_file)

    # 保存实验元数据
    current_config = Config() # 创建一个临时的 Config 实例来保存当前配置
    current_config.LOOKBACK_WINDOW = LOOKBACK_WINDOW
    current_config.PREDICTION_HORIZON = PREDICTION_HORIZONS[0] # 设置为第一个预测范围
    current_config.TEACHER_MODEL_NAME = MODELS[0][0]
    current_config.STUDENT_MODEL_NAME = MODELS[0][1]
    current_config.EPOCHS = EPOCHS
    current_config.STABILITY_RUNS = STABILITY_RUNS
    current_config.ROBUSTNESS_NOISE_LEVELS = NOISE_LEVELS
    current_config.SMOOTHING_FACTORS = SMOOTHING_FACTORS
    current_config.update_model_configs() # 确保模型配置是最新的
    
    # 保存实验总览文件到根目录
    metadata_path = os.path.join(base_results_dir, "experiment_overview.json")
    
    # 直接生成实验总览文件，而不是依赖save_experiment_metadata函数的默认命名
    metadata = {}
    # 遍历Config对象的所有属性
    for attr_name in dir(current_config):
        if not attr_name.startswith('__') and not callable(getattr(current_config, attr_name)):
            attr_value = getattr(current_config, attr_name)
            # 尝试将所有可序列化的属性添加到元数据中
            try:
                # 对于字典，进行深拷贝以避免修改原始配置
                if isinstance(attr_value, dict):
                    metadata[attr_name] = attr_value.copy()
                else:
                    metadata[attr_name] = attr_value
            except TypeError:
                # 如果属性不可序列化，则跳过或转换为字符串
                metadata[attr_name] = str(attr_value)
    
    # 移除一些不必要的或敏感的路径信息
    if 'BASE_DIR' in metadata:
        del metadata['BASE_DIR']
    if 'LOG_FILE_PATH' in metadata:
        del metadata['LOG_FILE_PATH']
    
    # 添加实验相关的额外信息
    metadata['DATASETS'] = list(DATASETS.keys())
    metadata['PREDICTION_HORIZONS'] = PREDICTION_HORIZONS
    metadata['MODELS'] = MODELS
    metadata['NOISE_LEVELS'] = NOISE_LEVELS
    metadata['SMOOTHING_FACTORS'] = SMOOTHING_FACTORS
    
    try:
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"实验总览保存到: {metadata_path}")
    except Exception as e:
        logger.error(f"保存实验总览时出错: {e}")

    all_experiment_results = []
    all_experiment_similarity_results = []

    logger.info("Starting comprehensive evaluation experiments...")

    total_subtasks_per_combination = (1 + len(NOISE_LEVELS) + len(SMOOTHING_FACTORS))
    total_experiments = len(DATASETS) * len(PREDICTION_HORIZONS) * len(MODELS) * total_subtasks_per_combination * STABILITY_RUNS
    with tqdm(total=total_experiments, desc="Overall Experiment Progress") as pbar:
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
                        experiment_type='standard', logger=logger,
                        base_output_dir=base_results_dir,
                        pbar=pbar # Pass pbar
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
                            experiment_type='noise_injection', logger=logger,
                            base_output_dir=base_results_dir,
                            pbar=pbar # Pass pbar
                        )
                        all_experiment_results.extend(run_results)
                        all_experiment_similarity_results.extend(sim_results)

                    # 3. 去噪平滑评估
                    logger.info("\n--- Running Denoising Smoothing Evaluation ---")
                    for smoothing_weight_smoothing in SMOOTHING_FACTORS:
                        run_results, sim_results = run_experiment(
                            dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                            teacher_model, student_model,
                            smoothing_weight_smoothing=smoothing_weight_smoothing, smoothing_method=SMOOTHING_METHOD,
                            experiment_type='denoising_smoothing', logger=logger,
                            base_output_dir=base_results_dir,
                            pbar=pbar # Pass pbar
                        )
                        all_experiment_results.extend(run_results)
                        all_experiment_similarity_results.extend(sim_results)
    
    logger.info("All experiments completed.")

    # 保存所有结果
    # 保存所有结果到根目录的CSV文件
    results_df = pd.DataFrame(all_experiment_results)
    similarity_df = pd.DataFrame(all_experiment_similarity_results)

    # 使用标准化的文件名
    results_csv_path = os.path.join(base_results_dir, "experiment_results.csv")
    similarity_csv_path = os.path.join(base_results_dir, "similarity_results.csv")

    save_results_to_csv(results_df, results_csv_path, logger)
    save_results_to_csv(similarity_df, similarity_csv_path, logger)

    logger.info(f"All experiment results saved to: {results_csv_path}")
    logger.info(f"All similarity results saved to: {similarity_csv_path}")

    # --- 可视化结果 ---
    logger.info("\n--- Generating Visualizations ---")

    # 噪音注入评估可视化
    plot_noise_evaluation(results_df, similarity_df, base_results_dir, logger)

    # 去噪平滑评估可视化
    plot_smoothing_evaluation(results_df, similarity_df, base_results_dir, logger)

    logger.info("All visualizations generated.")
    logger.info("Comprehensive evaluation experiments completed.")

def plot_noise_evaluation(results_df, similarity_df, output_dir, logger):
    noise_df = results_df[results_df['experiment_type'] == 'noise_injection']
    noise_sim_df = similarity_df[similarity_df['experiment_type'] == 'noise_injection']

    if noise_df.empty:
        logger.warning("No noise injection data to plot.")
        return

    metrics = ['MSE', 'MAE'] # 假设这些是评估指标
    models_to_plot = ['Teacher', 'TaskOnly', 'Follower', 'RDT']

    for dataset in noise_df['dataset'].unique():
        for horizon in noise_df['pred_horizon'].unique():
            subset_df = noise_df[(noise_df['dataset'] == dataset) & (noise_df['pred_horizon'] == horizon)]
            subset_sim_df = noise_sim_df[(noise_sim_df['dataset'] == dataset) & (noise_sim_df['pred_horizon'] == horizon)]

            if subset_df.empty:
                continue

            # Plot performance metrics vs. noise level
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                for model_prefix in models_to_plot:
                    col_name = f'{model_prefix}_{metric}'
                    if col_name in subset_df.columns:
                        # Calculate mean across stability runs
                        plot_data = subset_df.groupby('noise_level')[col_name].mean().reset_index()
                        plt.plot(plot_data['noise_level'], plot_data[col_name], marker='o', label=model_prefix)
                plt.title(f'{dataset} - Horizon {horizon} - {metric} vs. Noise Level')
                plt.xlabel('Noise Level')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True)
                save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_noise_{metric}.png'), logger)
                plt.close()

            # Plot similarity vs. noise level
            if not subset_sim_df.empty:
                plt.figure(figsize=(12, 6))
                for model_prefix in ['TaskOnly', 'Follower', 'RDT']: # 只有学生模型有相似度
                    col_name = f'{model_prefix}_Simi_cosine_similarity' # 假设相似度指标是cosine_similarity
                    if col_name in subset_sim_df.columns:
                        plot_data = subset_sim_df.groupby('noise_level')[col_name].mean().reset_index()
                        plt.plot(plot_data['noise_level'], plot_data[col_name], marker='o', label=model_prefix)
                plt.title(f'{dataset} - Horizon {horizon} - Student-Teacher Cosine Similarity vs. Noise Level')
                plt.xlabel('Noise Level')
                plt.ylabel('Cosine Similarity')
                plt.legend()
                plt.grid(True)
                save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_noise_similarity.png'), logger)
                plt.close()

def plot_smoothing_evaluation(results_df, similarity_df, output_dir, logger):
    smoothing_df = results_df[results_df['experiment_type'] == 'denoising_smoothing']
    smoothing_sim_df = similarity_df[similarity_df['experiment_type'] == 'denoising_smoothing']

    if smoothing_df.empty:
        logger.warning("No smoothing evaluation data to plot.")
        return

    metrics = ['MSE', 'MAE'] # 假设这些是评估指标
    models_to_plot = ['Teacher', 'TaskOnly', 'Follower', 'RDT']

    for dataset in smoothing_df['dataset'].unique():
        for horizon in smoothing_df['pred_horizon'].unique():
            subset_df = smoothing_df[(smoothing_df['dataset'] == dataset) & (smoothing_df['pred_horizon'] == horizon)]
            subset_sim_df = smoothing_sim_df[(smoothing_sim_df['dataset'] == dataset) & (smoothing_sim_df['pred_horizon'] == horizon)]

            if subset_df.empty:
                continue

            # Plot performance metrics vs. smoothing factor
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                for model_prefix in models_to_plot:
                    col_name = f'{model_prefix}_{metric}'
                    if col_name in subset_df.columns:
                        plot_data = subset_df.groupby('smoothing_factor')[col_name].mean().reset_index()
                        plt.plot(plot_data['smoothing_factor'], plot_data[col_name], marker='o', label=model_prefix)
                plt.title(f'{dataset} - Horizon {horizon} - {metric} vs. Smoothing Factor')
                plt.xlabel('Smoothing Factor')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True)
                save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_smoothing_{metric}.png'), logger)
                plt.close()

            # Plot similarity vs. smoothing factor
            if not subset_sim_df.empty:
                plt.figure(figsize=(12, 6))
                for model_prefix in ['TaskOnly', 'Follower', 'RDT']:
                    col_name = f'{model_prefix}_Simi_cosine_similarity'
                    if col_name in subset_sim_df.columns:
                        plot_data = subset_sim_df.groupby('smoothing_factor')[col_name].mean().reset_index()
                        plt.plot(plot_data['smoothing_factor'], plot_data[col_name], marker='o', label=model_prefix)
                plt.title(f'{dataset} - Horizon {horizon} - Student-Teacher Cosine Similarity vs. Smoothing Factor')
                plt.xlabel('Smoothing Factor')
                plt.ylabel('Cosine Similarity')
                plt.legend()
                plt.grid(True)
                save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_smoothing_similarity.png'), logger)
                plt.close()

if __name__ == "__main__":
    main()