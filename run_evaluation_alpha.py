# run_evaluation_alpha.py (Pseudocode)

# 导入必要的模块 (与 run_evaluation_experiments.py 相同)
import os
import json
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import gc  # 导入垃圾回收模块
import psutil  # 用于监控内存使用情况

from src.config import Config
from src.data_handler import TimeSeriesDataset, load_and_preprocess_data
from src.models import get_model
from src.trainer import StandardTrainer, RDT_Trainer, get_optimizer, get_loss_function
from src.schedulers import get_alpha_scheduler
from src.evaluator import evaluate_model, calculate_similarity_metrics
from src.utils import set_seed, setup_logging, save_results_to_csv, save_plot, save_experiment_metadata
import src.utils as utils

# 辅助函数：清理内存
def clean_memory(logger=None):
    """清理内存，释放未使用的缓存"""
    # 强制垃圾回收
    gc.collect()
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if logger:
        # 获取当前进程的内存使用情况
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"当前内存使用: {memory_info.rss / (1024 * 1024):.2f} MB")

# --- 实验配置 (与 run_evaluation_experiments.py 相同，或根据需要调整) ---
DATASETS = {
    # 'exchange_rate': './data/exchange_rate.csv',
    # 'national_illness': './data/national_illness.csv',
    'weather': './data/weather.csv',
    'ETTh1': './data/ETT-small/ETTh1.csv',
    # 'ETTh2': './data/ETT-small/ETTh2.csv',
}

PREDICTION_HORIZONS = [96, 336, 720]
LOOKBACK_WINDOW = 192
EPOCHS = 100
STABILITY_RUNS = 1

MODELS = [('DLinear', 'PatchTST')]

# NOISE_LEVELS = [ 0.02, 0.10, 0.20]
NOISE_LEVELS = []
NOISE_TYPE = 'gaussian'

# SMOOTHING_FACTORS = [ 0.10,  0.30, 0.5]
SMOOTHING_FACTORS = []
SMOOTHING_METHOD = 'moving_average'

# 新增固定 Alpha 值列表
FIXED_ALPHA_VALUES = [0.2, 0.4, 0.6, 0.8]

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
    experiment_type='standard',
    logger=None,
    base_output_dir=None,
    pbar=None
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
        
        # 修改：使用传入的 base_output_dir
        if base_output_dir is None:
            base_output_dir = "results_alpha" # 默认值，但 main 函数会传入
        
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

        # 加载和预处理数据 (与 run_evaluation_experiments.py 相同)
        config = Config()
        config.LOOKBACK_WINDOW = lookback_window
        config.PREDICTION_HORIZON = pred_horizon
        config.TEACHER_MODEL_NAME = teacher_model_name
        config.STUDENT_MODEL_NAME = student_model_name
        config.EPOCHS = epochs
        config.TRAIN_NOISE_INJECTION_LEVEL = noise_level
        config.NOISE_TYPE = noise_type
        config.SMOOTHING_FACTOR = 24
        config.SMOOTHING_WEIGHT_SMOOTHING = smoothing_weight_smoothing
        config.SMOOTHING_METHOD = smoothing_method
        config.SIMILARITY_METRIC = 'cosine_similarity'
        config.RESULTS_DIR = experiment_results_dir
        config.PLOTS_DIR = os.path.join(config.RESULTS_DIR, "plots")
        
        config.DATASET_NAME = dataset_name
        config.NOISE_LEVEL = noise_level
        config.SMOOTHING_WEIGHT = smoothing_weight_smoothing
        config.RUN_IDX = run_idx + 1
        config.SEED = current_seed

        if dataset_name == 'exchange_rate':
            config.EXOGENOUS_COLS = ['0', '1', '2', '3', '4', '5', '6']
            logger.info(f"Setting EXOGENOUS_COLS for {dataset_name}: {config.EXOGENOUS_COLS}")
        else:
            config.EXOGENOUS_COLS = []
            logger.info(f"Setting EXOGENOUS_COLS for {dataset_name}: {config.EXOGENOUS_COLS}")

        config.update_model_configs()

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, "plots"), exist_ok=True)

        if experiment_type == 'noise_injection':
            config.VAL_NOISE_INJECTION_LEVEL = noise_level
        elif experiment_type == 'denoising_smoothing':
            pass
        else:
            config.TRAIN_NOISE_INJECTION_LEVEL = 0
            config.NOISE_TYPE = None
            config.SMOOTHING_FACTOR = 0
            config.SMOOTHING_METHOD = None
            config.SMOOTHING_APPLY_TEST = False
        
        if experiment_type == 'denoising_smoothing':
            config.SMOOTHING_APPLY_TEST = True
            config.SMOOTHING_APPLY_TRAIN = True
            config.SMOOTHING_APPLY_VAL = True
        else:
            config.SMOOTHING_APPLY_TEST = False
            config.SMOOTHING_APPLY_TRAIN = False
            config.SMOOTHING_APPLY_VAL = False

        dataset_filename = os.path.basename(dataset_path)
        time_freq_for_dataset = config.DATASET_TIME_FREQ_MAP.get(dataset_filename, 'h')
        config.TIME_FREQ = time_freq_for_dataset
        logger.info(f"Using time frequency '{time_freq_for_dataset}' for dataset '{dataset_filename}'")

        train_loader, val_loader, test_loader, scaler, n_features = load_and_preprocess_data(
            dataset_path, config, logger, time_freq=time_freq_for_dataset
        )
        config.N_FEATURES = n_features
        config.update_model_configs()

        actual_device_str = config.DEVICE
        if "cuda" in actual_device_str.lower() and not torch.cuda.is_available():
            logger.warning(f"配置请求 CUDA ({actual_device_str}) 但 CUDA 不可用。将使用 CPU。")
            actual_device_str = "cpu"
        elif "cuda" in actual_device_str.lower() and torch.cuda.is_available():
            logger.info(f"CUDA 可用，将尝试使用设备: {actual_device_str}")
        else:
            actual_device_str = "cpu"
            logger.info(f"将使用 CPU。")

        device = torch.device(actual_device_str)
        config.DEVICE = str(device)
        logger.info(f"实验最终将在设备上运行: {device}")

        # 记录初始内存使用情况
        logger.info("开始实验前内存状态:")
        clean_memory(logger)
        
        teacher_model = get_model(teacher_model_name, config).to(device)
        student_model = get_model(student_model_name, config).to(device)

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
        
        # 保存训练历史记录用于绘图
        teacher_history = teacher_trainer.history.copy()
        
        # 清理训练过程中产生的中间变量
        del teacher_trainer
        clean_memory(logger)
        
        # 从磁盘加载最佳模型，而不是保留内存中的模型
        teacher_model = get_model(teacher_model_name, config).to(device)
        teacher_model = utils.load_model(teacher_model, os.path.join(config.RESULTS_DIR, f"{teacher_model_name}_teacher_model.pt"), config.DEVICE)

        # 评估 Teacher 模型
        teacher_metrics, teacher_true_original, teacher_preds_original = evaluate_model(
            teacher_model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=teacher_model_name,
            dataset_type="Test Set"
        )
        plot_save_dir = os.path.join(config.PLOTS_DIR, teacher_model_name.lower().replace(" ", "_"))
        utils.plot_training_metrics(teacher_history, plot_save_dir, teacher_model_name)
        utils.plot_weights_biases_distribution(teacher_model, plot_save_dir, teacher_model_name)
        for metric, value in teacher_metrics.items():
            results[f'Teacher_{metric}'] = value
            
        logger.info("Teacher模型训练和评估后内存状态:")
        clean_memory(logger)

        # 训练 TaskOnly 模型 (alpha=1)
        logger.info(f"Training TaskOnly Model (Student: {student_model_name}, alpha=1)")
        config.ALPHA_SCHEDULE = 'constant'
        config.CONSTANT_ALPHA = 1.0
        
        # 重新初始化学生模型，避免使用之前的模型状态
        student_model = get_model(student_model_name, config).to(device)
        
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
            config_obj=config,
            model_name=f"{student_model_name}_TaskOnly"
        )
        task_only_trainer.train()
        
        # 清理训练过程中产生的中间变量
        task_only_history = task_only_trainer.history.copy()  # 保存历史记录用于绘图
        del task_only_trainer
        clean_memory(logger)
        
        # 从磁盘加载最佳模型
        task_only_model = get_model(student_model_name, config).to(device)
        task_only_model = utils.load_model(task_only_model, os.path.join(config.RESULTS_DIR, f"{student_model_name}_task_only_model.pt"), config.DEVICE)

        # 评估 TaskOnly 模型
        task_only_metrics, _, task_only_preds_original = evaluate_model(
            task_only_model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=f"{student_model_name}_TaskOnly",
            teacher_predictions_original=teacher_preds_original,
            dataset_type="Test Set"
        )
        plot_save_dir = os.path.join(config.PLOTS_DIR, f"{student_model_name}_taskonly".lower().replace(" ", "_"))
        utils.plot_training_metrics(task_only_history, plot_save_dir, f"{student_model_name}_TaskOnly")
        utils.plot_weights_biases_distribution(task_only_model, plot_save_dir, f"{student_model_name}_TaskOnly")
        for metric, value in task_only_metrics.items():
            results[f'TaskOnly_{metric}'] = value
            
        # 释放不再需要的变量
        del task_only_model, task_only_history, task_only_preds_original
        logger.info("TaskOnly模型训练和评估后内存状态:")
        clean_memory(logger)

        # 训练 Follower 模型 (alpha=0)
        logger.info(f"Training Follower Model (Student: {student_model_name}, alpha=0)")
        config.ALPHA_SCHEDULE = 'constant'
        config.CONSTANT_ALPHA = 0.0
        
        # 重新初始化学生模型
        student_model = get_model(student_model_name, config).to(device)
        
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
            config_obj=config,
            model_name=f"{student_model_name}_Follower"
        )
        follower_trainer.train()
        
        # 清理训练过程中产生的中间变量
        follower_history = follower_trainer.history.copy()  # 保存历史记录用于绘图
        del follower_trainer
        clean_memory(logger)
        
        # 从磁盘加载最佳模型
        follower_model = get_model(student_model_name, config).to(device)
        follower_model = utils.load_model(follower_model, os.path.join(config.RESULTS_DIR, f"{student_model_name}_follower_model.pt"), config.DEVICE)

        # 评估 Follower 模型
        follower_metrics, _, follower_preds_original = evaluate_model(
            follower_model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=f"{student_model_name}_Follower",
            teacher_predictions_original=teacher_preds_original,
            dataset_type="Test Set"
        )
        plot_save_dir = os.path.join(config.PLOTS_DIR, f"{student_model_name}_follower".lower().replace(" ", "_"))
        utils.plot_training_metrics(follower_history, plot_save_dir, f"{student_model_name}_Follower")
        utils.plot_weights_biases_distribution(follower_model, plot_save_dir, f"{student_model_name}_Follower")
        for metric, value in follower_metrics.items():
            results[f'Follower_{metric}'] = value
            
        # 释放不再需要的变量
        del follower_model, follower_history, follower_preds_original
        logger.info("Follower模型训练和评估后内存状态:")
        clean_memory(logger)

        # 训练 RDT 模型 (alpha 动态调度)
        logger.info(f"Training RDT Model (Student: {student_model_name}, dynamic alpha)")
        config.ALPHA_SCHEDULE = 'linear'
        config.CONSTANT_ALPHA = None
        
        # 重新初始化学生模型
        student_model = get_model(student_model_name, config).to(device)
        
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
            config_obj=config,
            model_name=f"{student_model_name}_RDT"
        )
        rdt_trainer.train()
        
        # 清理训练过程中产生的中间变量
        rdt_history = rdt_trainer.history.copy()  # 保存历史记录用于绘图
        del rdt_trainer
        clean_memory(logger)
        
        # 从磁盘加载最佳模型
        rdt_model = get_model(student_model_name, config).to(device)
        rdt_model = utils.load_model(rdt_model, os.path.join(config.RESULTS_DIR, f"{student_model_name}_rdt_model.pt"), config.DEVICE)

        # 评估 RDT 模型
        rdt_metrics, _, rdt_preds_original = evaluate_model(
            rdt_model, test_loader, config.DEVICE, scaler, config, logger,
            model_name=f"{student_model_name}_RDT",
            teacher_predictions_original=teacher_preds_original,
            dataset_type="Test Set"
        )
        plot_save_dir = os.path.join(config.PLOTS_DIR, f"{student_model_name}_rdt".lower().replace(" ", "_"))
        utils.plot_training_metrics(rdt_history, plot_save_dir, f"{student_model_name}_RDT")
        utils.plot_weights_biases_distribution(rdt_model, plot_save_dir, f"{student_model_name}_RDT")
        for metric, value in rdt_metrics.items():
            results[f'RDT_{metric}'] = value
            
        # 释放不再需要的变量
        del rdt_model, rdt_history, rdt_preds_original
        logger.info("RDT模型训练和评估后内存状态:")
        clean_memory(logger)

        # --- 新增：训练和评估固定 Alpha 值的模型 ---
        for alpha_val in FIXED_ALPHA_VALUES:
            model_name_fixed_alpha = f"{student_model_name}_Alpha{str(alpha_val).replace('.', '')}"
            logger.info(f"Training Fixed Alpha Model (Student: {student_model_name}, alpha={alpha_val})")
            
            # 重新初始化 config，避免影响后续实验
            current_fixed_alpha_config = Config()
            current_fixed_alpha_config.LOOKBACK_WINDOW = lookback_window
            current_fixed_alpha_config.PREDICTION_HORIZON = pred_horizon
            current_fixed_alpha_config.TEACHER_MODEL_NAME = teacher_model_name
            current_fixed_alpha_config.STUDENT_MODEL_NAME = student_model_name
            current_fixed_alpha_config.EPOCHS = epochs
            current_fixed_alpha_config.TRAIN_NOISE_INJECTION_LEVEL = noise_level
            current_fixed_alpha_config.NOISE_TYPE = noise_type
            current_fixed_alpha_config.SMOOTHING_FACTOR = 24
            current_fixed_alpha_config.SMOOTHING_WEIGHT_SMOOTHING = smoothing_weight_smoothing
            current_fixed_alpha_config.SMOOTHING_METHOD = smoothing_method
            current_fixed_alpha_config.SIMILARITY_METRIC = 'cosine_similarity'
            current_fixed_alpha_config.RESULTS_DIR = experiment_results_dir # 仍然使用当前实验的results目录
            current_fixed_alpha_config.PLOTS_DIR = os.path.join(current_fixed_alpha_config.RESULTS_DIR, "plots")
            
            current_fixed_alpha_config.DATASET_NAME = dataset_name
            current_fixed_alpha_config.NOISE_LEVEL = noise_level
            current_fixed_alpha_config.SMOOTHING_WEIGHT = smoothing_weight_smoothing
            current_fixed_alpha_config.RUN_IDX = run_idx + 1
            current_fixed_alpha_config.SEED = current_seed

            if dataset_name == 'exchange_rate':
                current_fixed_alpha_config.EXOGENOUS_COLS = ['0', '1', '2', '3', '4', '5', '6']
            else:
                current_fixed_alpha_config.EXOGENOUS_COLS = []
            current_fixed_alpha_config.update_model_configs()

            if experiment_type == 'noise_injection':
                current_fixed_alpha_config.VAL_NOISE_INJECTION_LEVEL = noise_level
            elif experiment_type == 'denoising_smoothing':
                pass
            else:
                current_fixed_alpha_config.TRAIN_NOISE_INJECTION_LEVEL = 0
                current_fixed_alpha_config.NOISE_TYPE = None
                current_fixed_alpha_config.SMOOTHING_FACTOR = 0
                current_fixed_alpha_config.SMOOTHING_METHOD = None
                current_fixed_alpha_config.SMOOTHING_APPLY_TEST = False
            
            if experiment_type == 'denoising_smoothing':
                current_fixed_alpha_config.SMOOTHING_APPLY_TEST = True
                current_fixed_alpha_config.SMOOTHING_APPLY_TRAIN = True
                current_fixed_alpha_config.SMOOTHING_APPLY_VAL = True
            else:
                current_fixed_alpha_config.SMOOTHING_APPLY_TEST = False
                current_fixed_alpha_config.SMOOTHING_APPLY_TRAIN = False
                current_fixed_alpha_config.SMOOTHING_APPLY_VAL = False

            # 确保使用正确的设备
            current_fixed_alpha_config.DEVICE = str(device) # 使用之前确定的设备

            # 设置固定 Alpha 值
            current_fixed_alpha_config.ALPHA_SCHEDULE = 'constant'
            current_fixed_alpha_config.CONSTANT_ALPHA = alpha_val

            # 重新获取学生模型实例，确保是新的模型，避免权重污染
            fixed_alpha_student_model = get_model(student_model_name, current_fixed_alpha_config).to(device)

            # 记录当前固定Alpha值模型训练开始前的内存状态
            logger.info(f"固定Alpha={alpha_val}模型训练前内存状态:")
            clean_memory(logger)
            
            fixed_alpha_trainer = RDT_Trainer(
                teacher_model=teacher_model,
                student_model=fixed_alpha_student_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=get_optimizer(fixed_alpha_student_model, current_fixed_alpha_config),
                task_loss_fn=get_loss_function(current_fixed_alpha_config),
                distill_loss_fn=get_loss_function(current_fixed_alpha_config),
                alpha_scheduler=get_alpha_scheduler(current_fixed_alpha_config),
                device=current_fixed_alpha_config.DEVICE,
                epochs=current_fixed_alpha_config.EPOCHS,
                patience=current_fixed_alpha_config.PATIENCE,
                model_save_path=os.path.join(current_fixed_alpha_config.RESULTS_DIR, f"{model_name_fixed_alpha}_model.pt"),
                scaler=scaler,
                config_obj=current_fixed_alpha_config,
                model_name=model_name_fixed_alpha
            )
            fixed_alpha_trainer.train()
            
            # 清理训练过程中产生的中间变量
            fixed_alpha_history = fixed_alpha_trainer.history.copy()  # 保存历史记录用于绘图
            del fixed_alpha_trainer
            clean_memory(logger)
            
            # 从磁盘加载最佳模型
            fixed_alpha_model = get_model(student_model_name, current_fixed_alpha_config).to(device)
            fixed_alpha_model = utils.load_model(fixed_alpha_model, os.path.join(current_fixed_alpha_config.RESULTS_DIR, f"{model_name_fixed_alpha}_model.pt"), current_fixed_alpha_config.DEVICE)

            fixed_alpha_metrics, _, fixed_alpha_preds_original = evaluate_model(
                fixed_alpha_model, test_loader, current_fixed_alpha_config.DEVICE, scaler, current_fixed_alpha_config, logger,
                model_name=model_name_fixed_alpha,
                teacher_predictions_original=teacher_preds_original,
                dataset_type="Test Set"
            )
            plot_save_dir = os.path.join(current_fixed_alpha_config.PLOTS_DIR, model_name_fixed_alpha.lower().replace(" ", "_"))
            utils.plot_training_metrics(fixed_alpha_history, plot_save_dir, model_name_fixed_alpha)
            utils.plot_weights_biases_distribution(fixed_alpha_model, plot_save_dir, model_name_fixed_alpha)
            for metric, value in fixed_alpha_metrics.items():
                results[f'{model_name_fixed_alpha}_{metric}'] = value
                
            # 释放不再需要的变量
            del fixed_alpha_model, fixed_alpha_history, fixed_alpha_preds_original, fixed_alpha_student_model, current_fixed_alpha_config
            logger.info(f"固定Alpha={alpha_val}模型训练和评估后内存状态:")
            clean_memory(logger)
        # --- 新增部分结束 ---

        # 从指标字典中提取相似度结果 (需要包含新增的固定 Alpha 模型)
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
        
        # 遍历 FIXED_ALPHA_VALUES，添加其相似度结果
        for alpha_val in FIXED_ALPHA_VALUES:
            model_name_fixed_alpha = f"{student_model_name}_Alpha{str(alpha_val).replace('.', '')}"
            if f'{model_name_fixed_alpha}_Simi_cosine_similarity' in results:
                similarity_results[f'{model_name_fixed_alpha}_Simi_cosine_similarity'] = results[f'{model_name_fixed_alpha}_Simi_cosine_similarity']


        # 记录当前运行的元数据 (需要包含新增的固定 Alpha 模型)
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

        def convert_floats(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64) or isinstance(obj, float):
                if np.isnan(obj):
                    return 1.0
                return float(obj)
            elif isinstance(obj, np.integer):  # 也处理整数类型
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_floats(elem) for elem in obj]
            return obj

        serializable_results = convert_floats(results)
        serializable_similarity_results = convert_floats(similarity_results)

        # 移除Teacher模型的相似度指标字段
        if 'Teacher_similarity_cosine_similarity' in serializable_results:
            del serializable_results['Teacher_similarity_cosine_similarity']
        if 'Teacher_error_cos_similarity' in serializable_results:
            del serializable_results['Teacher_error_cos_similarity']

        run_metrics_path = os.path.join(experiment_results_dir, "run_metrics.json")
        with open(run_metrics_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Individual run metrics saved to: {run_metrics_path}")

        # 同样移除similarity_results中的Teacher相似度指标
        if 'Teacher_similarity_cosine_similarity' in serializable_similarity_results:
            del serializable_similarity_results['Teacher_similarity_cosine_similarity']
        if 'Teacher_error_cos_similarity' in serializable_similarity_results:
            del serializable_similarity_results['Teacher_error_cos_similarity']

        run_similarity_path = os.path.join(experiment_results_dir, "run_similarity.json")
        with open(run_similarity_path, 'w') as f:
            json.dump(serializable_similarity_results, f, indent=4)
        logger.info(f"Individual run similarity results saved to: {run_similarity_path}")

        with open(completion_marker_file, 'w') as f:
            f.write("completed")
        logger.info(f"--- Stability Run {run_idx + 1}/{stability_runs} (Seed: {current_seed}) completed and marked. ---")
        
        if pbar:
            pbar.update(1)

    return all_run_results, all_similarity_results

def main():
    # 修改：将结果保存到 results_alpha 文件夹
    base_results_dir = "results_alpha"
    os.makedirs(base_results_dir, exist_ok=True)

    os.makedirs("log", exist_ok=True)
    log_file = os.path.join("log", "experiment_alpha.log") # 可以修改日志文件名
    logger = setup_logging(log_file)
    
    # 初始化时清理内存
    logger.info("实验开始前内存状态:")
    clean_memory(logger)

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
    
    metadata_path = os.path.join(base_results_dir, "experiment_overview_alpha.json") # 修改元数据文件名
    
    metadata = {}
    for attr_name in dir(current_config):
        if not attr_name.startswith('__') and not callable(getattr(current_config, attr_name)):
            attr_value = getattr(current_config, attr_name)
            try:
                if isinstance(attr_value, dict):
                    metadata[attr_name] = attr_value.copy()
                else:
                    metadata[attr_name] = attr_value
            except TypeError:
                metadata[attr_name] = str(attr_value)
    
    if 'BASE_DIR' in metadata:
        del metadata['BASE_DIR']
    if 'LOG_FILE_PATH' in metadata:
        del metadata['LOG_FILE_PATH']
    
    metadata['DATASETS'] = list(DATASETS.keys())
    metadata['PREDICTION_HORIZONS'] = PREDICTION_HORIZONS
    metadata['MODELS'] = MODELS
    metadata['NOISE_LEVELS'] = NOISE_LEVELS
    metadata['SMOOTHING_FACTORS'] = SMOOTHING_FACTORS
    metadata['FIXED_ALPHA_VALUES'] = FIXED_ALPHA_VALUES # 新增：记录固定 Alpha 值

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

    # 调整 total_subtasks_per_combination 以包含新的固定 Alpha 模型
    total_subtasks_per_combination = (1 + len(NOISE_LEVELS) + len(SMOOTHING_FACTORS) + len(FIXED_ALPHA_VALUES))
    total_experiments = len(DATASETS) * len(PREDICTION_HORIZONS) * len(MODELS) * total_subtasks_per_combination * STABILITY_RUNS
    
    # 在每个主要实验阶段之间清理内存
    def clean_between_experiments():
        # 强制垃圾回收
        gc.collect()
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("实验阶段之间内存清理完成")
    with tqdm(total=total_experiments, desc="Overall Experiment Progress") as pbar:
        for dataset_name, dataset_path in DATASETS.items():
            for pred_horizon in PREDICTION_HORIZONS:
                for teacher_model, student_model in MODELS:
                    logger.info(f"\n--- Running for Dataset: {dataset_name}, Horizon: {pred_horizon}, "
                                f"Models: {teacher_model}/{student_model} ---")

                    teacher_name_for_dir = teacher_model if teacher_model else "NoTeacher"
                    current_experiment_combination_dir = (
                        f"{dataset_name}_{teacher_name_for_dir}_{student_model}_"
                        f"h{pred_horizon}_noise0_smooth_w0.0"
                    )
                    # 修改：使用新的 base_results_dir
                    current_base_output_dir = os.path.join(base_results_dir, current_experiment_combination_dir)
                    os.makedirs(current_base_output_dir, exist_ok=True)

                    # 1. 标准评估 (无噪音, 无平滑)
                    logger.info("\n--- Running Standard Evaluation ---")
                    run_results, sim_results = run_experiment(
                        dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                        teacher_model, student_model,
                        noise_level=0, smoothing_weight_smoothing=0.0,
                        experiment_type='standard', logger=logger,
                        base_output_dir=base_results_dir, # 传递新的 base_results_dir
                        pbar=pbar
                    )
                    all_experiment_results.extend(run_results)
                    all_experiment_similarity_results.extend(sim_results)
                    
                    # 在不同实验类型之间清理内存
                    clean_between_experiments()

                    # 2. 噪音注入评估
                    logger.info("\n--- Running Noise Injection Evaluation ---")
                    for noise_level in NOISE_LEVELS:
                        run_results, sim_results = run_experiment(
                            dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                            teacher_model, student_model,
                            noise_level=noise_level, noise_type=NOISE_TYPE, smoothing_weight_smoothing=0.0,
                            experiment_type='noise_injection', logger=logger,
                            base_output_dir=base_results_dir, # 传递新的 base_results_dir
                            pbar=pbar
                        )
                        all_experiment_results.extend(run_results)
                        all_experiment_similarity_results.extend(sim_results)
                    
                    # 在不同实验类型之间清理内存
                    clean_between_experiments()

                    # 3. 去噪平滑评估
                    logger.info("\n--- Running Denoising Smoothing Evaluation ---")
                    for smoothing_weight_smoothing in SMOOTHING_FACTORS:
                        run_results, sim_results = run_experiment(
                            dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                            teacher_model, student_model,
                            noise_level=0,
                            smoothing_weight_smoothing=smoothing_weight_smoothing, smoothing_method=SMOOTHING_METHOD,
                            experiment_type='denoising_smoothing', logger=logger,
                            base_output_dir=base_results_dir, # 传递新的 base_results_dir
                            pbar=pbar
                        )
                        all_experiment_results.extend(run_results)
                        all_experiment_similarity_results.extend(sim_results)
                    
                    # 在不同实验类型之间清理内存
                    clean_between_experiments()
                    
                    # 在每个实验组合完成后，生成并保存可视化结果
                    logger.info(f"\n--- Generating Visualizations for {current_experiment_combination_dir} ---")
                    
                    current_combo_results_df = pd.DataFrame(all_experiment_results)
                    current_combo_sim_df = pd.DataFrame(all_experiment_similarity_results)

                    plot_noise_evaluation(current_combo_results_df, current_combo_sim_df, current_base_output_dir, logger)
                    plot_smoothing_evaluation(current_combo_results_df, current_combo_sim_df, current_base_output_dir, logger)
                    # 新增：绘制固定 Alpha 值的评估结果
                    plot_fixed_alpha_evaluation(current_combo_results_df, current_combo_sim_df, current_base_output_dir, logger)

                    logger.info(f"Visualizations generated for {current_experiment_combination_dir}.")

    logger.info("All experiments completed.")

    results_df = pd.DataFrame(all_experiment_results)
    similarity_df = pd.DataFrame(all_experiment_similarity_results)

    results_csv_path = os.path.join(base_results_dir, "experiment_results.csv")
    similarity_csv_path = os.path.join(base_results_dir, "similarity_results.csv")

    save_results_to_csv(results_df, results_csv_path, logger)
    save_results_to_csv(similarity_df, similarity_csv_path, logger)

    logger.info(f"All experiment results saved to: {results_csv_path}")
    logger.info(f"All similarity results saved to: {similarity_csv_path}")

    # 实验结束后最终清理内存
    logger.info("实验结束后内存状态:")
    clean_memory(logger)
    
    logger.info("Comprehensive evaluation experiments completed.")

# --- 新增绘图函数：plot_fixed_alpha_evaluation ---
def plot_fixed_alpha_evaluation(results_df, similarity_df, output_dir, logger):
    # 识别固定 Alpha 模型的前缀
    fixed_alpha_model_prefixes = [f'Alpha{str(alpha).replace(".", "")}' for alpha in FIXED_ALPHA_VALUES]
    
    # 过滤出包含固定 Alpha 模型结果的行
    # 检查 results_df 的列是否包含固定 Alpha 模型的前缀
    has_fixed_alpha_cols = False
    for col in results_df.columns:
        for prefix in fixed_alpha_model_prefixes:
            if col.startswith(f'PatchTST_{prefix}'):
                has_fixed_alpha_cols = True
                break
        if has_fixed_alpha_cols:
            break
    
    if has_fixed_alpha_cols:
        # 如果有固定 Alpha 模型的列，就使用原始的 results_df
        fixed_alpha_df = results_df.copy()
    else:
        # 如果没有固定 Alpha 模型的列，就创建一个空的 DataFrame
        fixed_alpha_df = pd.DataFrame()
    
    # 同样处理相似度数据
    # 检查 similarity_df 的列是否包含固定 Alpha 模型的前缀
    has_fixed_alpha_cols = False
    for col in similarity_df.columns:
        for prefix in fixed_alpha_model_prefixes:
            if col.startswith(f'PatchTST_{prefix}'):
                has_fixed_alpha_cols = True
                break
        if has_fixed_alpha_cols:
            break
    
    if has_fixed_alpha_cols:
        # 如果有固定 Alpha 模型的列，就使用原始的 similarity_df
        fixed_alpha_sim_df = similarity_df.copy()
    else:
        # 如果没有固定 Alpha 模型的列，就创建一个空的 DataFrame
        fixed_alpha_sim_df = pd.DataFrame()

    if fixed_alpha_df.empty:
        logger.warning("No fixed alpha evaluation data to plot.")
        return

    metrics = ['MSE', 'MAE']
    models_to_plot = ['Teacher', 'TaskOnly', 'Follower', 'RDT'] + [f'Alpha{str(alpha).replace(".", "")}' for alpha in FIXED_ALPHA_VALUES]

    for dataset in fixed_alpha_df['dataset'].unique():
        for horizon in fixed_alpha_df['pred_horizon'].unique():
            subset_df = fixed_alpha_df[(fixed_alpha_df['dataset'] == dataset) & (fixed_alpha_df['pred_horizon'] == horizon)]
            subset_sim_df = fixed_alpha_sim_df[(fixed_alpha_sim_df['dataset'] == dataset) & (fixed_alpha_sim_df['pred_horizon'] == horizon)]

            if subset_df.empty:
                continue

            # Plot performance metrics vs. Alpha value
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                plot_data_points = []
                for model_prefix in models_to_plot:
                    if model_prefix == 'Teacher':
                        pass
                    elif model_prefix == 'TaskOnly':
                        col_name = f'TaskOnly_{metric}'
                        if col_name in subset_df.columns:
                            plot_data_points.append({'alpha': 1.0, 'metric_value': subset_df[col_name].mean()})
                    elif model_prefix == 'Follower':
                        col_name = f'Follower_{metric}'
                        if col_name in subset_df.columns:
                            plot_data_points.append({'alpha': 0.0, 'metric_value': subset_df[col_name].mean()})
                    elif model_prefix == 'RDT':
                        pass
                    else:
                        alpha_val_str = model_prefix.replace('Alpha', '')
                        alpha_val = float(alpha_val_str[0] + '.' + alpha_val_str[1:])
                        col_name = f'PatchTST_{model_prefix}_{metric}'
                        if col_name in subset_df.columns:
                            plot_data_points.append({'alpha': alpha_val, 'metric_value': subset_df[col_name].mean()})
                
                plot_data_points.sort(key=lambda x: x['alpha'])
                alphas = [p['alpha'] for p in plot_data_points]
                metric_values = [p['metric_value'] for p in plot_data_points]

                plt.plot(alphas, metric_values, marker='o', label='Student Models (Fixed Alpha)')
                
                if f'Teacher_{metric}' in subset_df.columns:
                    teacher_mean = subset_df[f'Teacher_{metric}'].mean()
                    plt.axhline(y=teacher_mean, color='r', linestyle='--', label=f'Teacher ({teacher_mean:.4f})')
                if f'RDT_{metric}' in subset_df.columns:
                    rdt_mean = subset_df[f'RDT_{metric}'].mean()
                    plt.axhline(y=rdt_mean, color='g', linestyle=':', label=f'RDT ({rdt_mean:.4f})')

                plt.title(f'{dataset} - Horizon {horizon} - {metric} vs. Alpha Value')
                plt.xlabel('Alpha Value')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True)
                utils.save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_fixed_alpha_{metric}.png'), logger)
                plt.close()

            # Plot similarity vs. Alpha value
            if not subset_sim_df.empty:
                plt.figure(figsize=(12, 6))
                plot_data_points_sim = []
                for model_prefix in ['TaskOnly', 'Follower'] + [f'Alpha{str(alpha).replace(".", "")}' for alpha in FIXED_ALPHA_VALUES]:
                    if model_prefix == 'TaskOnly':
                        col_name = f'TaskOnly_Simi_cosine_similarity'
                        if col_name in subset_sim_df.columns:
                            plot_data_points_sim.append({'alpha': 1.0, 'similarity_value': subset_sim_df[col_name].mean()})
                    elif model_prefix == 'Follower':
                        col_name = f'Follower_Simi_cosine_similarity'
                        if col_name in subset_sim_df.columns:
                            plot_data_points_sim.append({'alpha': 0.0, 'similarity_value': subset_sim_df[col_name].mean()})
                    else:
                        alpha_val_str = model_prefix.replace('Alpha', '')
                        alpha_val = float(alpha_val_str[0] + '.' + alpha_val_str[1:])
                        col_name = f'PatchTST_{model_prefix}_Simi_cosine_similarity'
                        if col_name in subset_sim_df.columns:
                            plot_data_points_sim.append({'alpha': alpha_val, 'similarity_value': subset_sim_df[col_name].mean()})
                
                plot_data_points_sim.sort(key=lambda x: x['alpha'])
                alphas_sim = [p['alpha'] for p in plot_data_points_sim]
                similarity_values = [p['similarity_value'] for p in plot_data_points_sim]

                plt.plot(alphas_sim, similarity_values, marker='o', label='Student Models (Fixed Alpha)')
                
                if f'RDT_Simi_cosine_similarity' in subset_sim_df.columns:
                    rdt_sim_mean = subset_sim_df[f'RDT_Simi_cosine_similarity'].mean()
                    plt.axhline(y=rdt_sim_mean, color='g', linestyle=':', label=f'RDT ({rdt_sim_mean:.4f})')

                plt.title(f'{dataset} - Horizon {horizon} - Student-Teacher Cosine Similarity vs. Alpha Value')
                plt.xlabel('Alpha Value')
                plt.ylabel('Cosine Similarity')
                plt.legend()
                plt.grid(True)
                utils.save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_fixed_alpha_similarity.png'), logger)
                plt.close()

# (plot_noise_evaluation 和 plot_smoothing_evaluation 函数保持不变)
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
                utils.save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_noise_{metric}.png'), logger)
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
                utils.save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_noise_similarity.png'), logger)
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
                utils.save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_smoothing_{metric}.png'), logger)
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
                utils.save_plot(plt, os.path.join(output_dir, f'{dataset}_{horizon}_smoothing_similarity.png'), logger)
                plt.close()
                
if __name__ == "__main__":
    main()