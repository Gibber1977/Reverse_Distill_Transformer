import torch
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import logging # 新增导入

# --- 导入自定义模块 ---
from src import config as default_config
from src import utils
from src.utils import setup_logging # 明确导入 setup_logging
from src.data_handler import load_and_preprocess_data
from src.models import get_teacher_model, get_student_model
from src.trainer import StandardTrainer, RDT_Trainer, get_optimizer, get_loss_function
from src.schedulers import get_alpha_scheduler, ConstantScheduler
from src.evaluator import evaluate_robustness, calculate_metrics, predict


def run_single_experiment(cfg, run_id, models_dir, plots_dir, metrics_dir):
    """
    运行单次完整的实验流程（用于稳定性评估）。
    评估所有成功训练的模型在 train, val, test 集上的指标。
    处理 cfg.TEACHER_MODEL_NAME 为 None 的情况 (TaskOnly)。
    """
    print(f"\n===== Starting Experiment Run {run_id + 1} / {cfg.STABILITY_RUNS} with Seed {cfg.SEED + run_id} =====")
    current_seed = cfg.SEED + run_id
    utils.set_seed(current_seed)

    model_paths = {} # 存储本次运行的模型路径
    all_split_metrics = {} # 存储所有模型在所有划分上的指标: {'ModelName': {'train': {...}, 'val': {...}, 'test': {...}}}

    # --- 创建本次运行的子目录 ---
    plots_run_dir = os.path.join(plots_dir, f'run_{run_id}')
    metrics_run_dir = os.path.join(metrics_dir, f'run_{run_id}')
    os.makedirs(plots_run_dir, exist_ok=True)
    os.makedirs(metrics_run_dir, exist_ok=True)

    # --- 1. 数据加载与预处理 ---
    try:
        train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(cfg)
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader} # Group loaders
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return None # 无法继续

    # --- 2. 初始化模型、优化器、损失函数 ---
    teacher_model = None
    trained_teacher_model = None
    if cfg.TEACHER_MODEL_NAME:
        print(f"\n--- Initializing Teacher Model: {cfg.TEACHER_MODEL_NAME} ---")
        teacher_model = get_teacher_model(cfg)
    else:
        print("\n--- No Teacher Model specified (TaskOnly mode) ---")

    print(f"\n--- Initializing Student Model: {cfg.STUDENT_MODEL_NAME} ---")
    # Student model instances will be created within training blocks

    task_loss_fn = get_loss_function(cfg)
    distill_loss_fn = get_loss_function(cfg) if cfg.TEACHER_MODEL_NAME else None # Only needed if teacher exists

    # --- 3. 训练教师模型 (Standard Training, if specified) ---
    if teacher_model:
        print(f"\n--- Training Teacher Model ({cfg.TEACHER_MODEL_NAME}) ---")
        teacher_optimizer = get_optimizer(teacher_model, cfg)
        teacher_model_save_path = os.path.join(models_dir, f"teacher_{cfg.TEACHER_MODEL_NAME}_run{run_id}_seed{current_seed}.pt")
        model_paths['teacher'] = teacher_model_save_path
        teacher_trainer = StandardTrainer(
            model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=teacher_optimizer,
            loss_fn=task_loss_fn,
            device=cfg.DEVICE,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            model_save_path=teacher_model_save_path,
            model_name=f"Teacher ({cfg.TEACHER_MODEL_NAME})"
        )
        trained_teacher_model, teacher_history = teacher_trainer.train()
        if trained_teacher_model:
             utils.plot_losses(teacher_history['train_loss'], teacher_history['val_loss'],
                               title=f"Teacher ({cfg.TEACHER_MODEL_NAME}) Training Loss (Run {run_id})",
                               save_path=os.path.join(plots_run_dir, f"teacher_loss.png"))
        else:
             print(f"Teacher model training failed or stopped early for run {run_id}. Skipping subsequent dependent steps.")
             # Allow continuing to train the student task-only model

    # --- 4. 训练学生模型 (Standard Training, Task Only) ---
    print(f"\n--- Training Student Model ({cfg.STUDENT_MODEL_NAME}) - Task Only ---")
    student_task_only_model = get_student_model(cfg)
    trained_student_task_only_model = None
    student_task_only_optimizer = get_optimizer(student_task_only_model, cfg)
    student_task_only_save_path = os.path.join(models_dir, f"student_{cfg.STUDENT_MODEL_NAME}_task_only_run{run_id}_seed{current_seed}.pt")
    model_paths['student_task_only'] = student_task_only_save_path
    student_task_only_trainer = StandardTrainer(
        model=student_task_only_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=student_task_only_optimizer,
        loss_fn=task_loss_fn,
        device=cfg.DEVICE,
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        model_save_path=student_task_only_save_path,
        model_name=f"Student ({cfg.STUDENT_MODEL_NAME}) Task Only"
    )
    trained_student_task_only_model, student_task_only_history = student_task_only_trainer.train()
    if trained_student_task_only_model:
        utils.plot_losses(student_task_only_history['train_loss'], student_task_only_history['val_loss'],
                          title=f"Student Task Only Training Loss (Run {run_id})",
                          save_path=os.path.join(plots_run_dir, f"student_task_only_loss.png"))
    else:
        print(f"Student Task Only model training failed or stopped early for run {run_id}.")
        # If this fails, subsequent steps might be less meaningful, but we continue evaluation

    # --- 5. 训练 RDT 学生模型 (if teacher exists and was trained) ---
    trained_student_rdt_model = None
    if trained_teacher_model and distill_loss_fn:
        print(f"\n--- Training RDT Student Model ({cfg.STUDENT_MODEL_NAME}) ---")
        student_rdt_model = get_student_model(cfg)
        student_rdt_optimizer = get_optimizer(student_rdt_model, cfg)
        alpha_scheduler = get_alpha_scheduler(cfg)
        student_rdt_save_path = os.path.join(models_dir, f"student_{cfg.STUDENT_MODEL_NAME}_rdt_run{run_id}_seed{current_seed}.pt")
        model_paths['student_rdt'] = student_rdt_save_path
        rdt_trainer = RDT_Trainer(
            student_model=student_rdt_model,
            teacher_model=trained_teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=student_rdt_optimizer,
            task_loss_fn=task_loss_fn,
            distill_loss_fn=distill_loss_fn,
            alpha_scheduler=alpha_scheduler,
            device=cfg.DEVICE,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            model_save_path=student_rdt_save_path,
            model_name=f"Student ({cfg.STUDENT_MODEL_NAME}) RDT"
        )
        trained_student_rdt_model, student_rdt_history = rdt_trainer.train()
        if trained_student_rdt_model:
            utils.plot_losses(student_rdt_history['train_loss'], student_rdt_history['val_loss'],
                              title=f"RDT Student Training Loss (Total Train, Task Val) (Run {run_id})",
                              save_path=os.path.join(plots_run_dir, f"student_rdt_loss.png"))
            if 'alpha' in student_rdt_history and student_rdt_history['alpha']:
                plt.figure()
                plt.plot(student_rdt_history['alpha'])
                plt.title(f"RDT Alpha Schedule (Run {run_id})")
                plt.xlabel("Epoch"); plt.ylabel("Alpha"); plt.grid(True)
                plt.savefig(os.path.join(plots_run_dir, f"student_rdt_alpha.png")); plt.close()
        else:
            print(f"Student RDT model training failed or stopped early for run {run_id}.")

    # --- 6. 训练追随者学生模型 (Follower, Alpha=0.0, if teacher exists and was trained) ---
    trained_student_follower_model = None
    if trained_teacher_model and distill_loss_fn:
        print(f"\n--- Training Follower Student Model ({cfg.STUDENT_MODEL_NAME}) (Distill Only, Alpha=0.0) ---")
        student_follower_model = get_student_model(cfg)
        student_follower_optimizer = get_optimizer(student_follower_model, cfg)
        follower_alpha_scheduler = ConstantScheduler(alpha_value=0.0, total_epochs=cfg.EPOCHS)
        student_follower_save_path = os.path.join(models_dir, f"student_{cfg.STUDENT_MODEL_NAME}_follower_run{run_id}_seed{current_seed}.pt")
        model_paths['student_follower'] = student_follower_save_path
        follower_trainer = RDT_Trainer(
            student_model=student_follower_model,
            teacher_model=trained_teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=student_follower_optimizer,
            task_loss_fn=task_loss_fn,
            distill_loss_fn=distill_loss_fn,
            alpha_scheduler=follower_alpha_scheduler,
            device=cfg.DEVICE,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            model_save_path=student_follower_save_path,
            model_name=f"Student ({cfg.STUDENT_MODEL_NAME}) Follower (Alpha=0.0)"
        )
        trained_student_follower_model, student_follower_history = follower_trainer.train()
        if trained_student_follower_model:
            utils.plot_losses(student_follower_history['train_loss'], student_follower_history['val_loss'],
                               title=f"Follower Student Training Loss (Distill Train, Task Val) (Run {run_id})",
                               save_path=os.path.join(plots_run_dir, f"student_follower_loss.png"))
        else:
             print(f"Student Follower model training failed or stopped early for run {run_id}.")

    # --- 7. 评估所有成功训练的模型 (在 Train, Val, Test 上) ---
    models_to_evaluate = {}
    if trained_teacher_model:
        models_to_evaluate["Teacher"] = trained_teacher_model
    if trained_student_task_only_model:
        # Use the standard name "TaskOnly" if teacher exists, otherwise just the student name
        eval_name = "Student_TaskOnly" if cfg.TEACHER_MODEL_NAME else cfg.STUDENT_MODEL_NAME
        models_to_evaluate[eval_name] = trained_student_task_only_model
        if 'student_task_only' in model_paths: # Check if key exists before popping
             model_paths[eval_name] = model_paths.pop('student_task_only') # Rename path key
    if trained_student_rdt_model:
        models_to_evaluate["Student_RDT"] = trained_student_rdt_model
    if trained_student_follower_model:
        models_to_evaluate["Student_Follower"] = trained_student_follower_model

    all_robustness_dfs = {} # Keep robustness evaluation separate

    print("\n--- Evaluating Models on Train, Validation, and Test Sets ---")

    for name, model in models_to_evaluate.items():
        print(f"\n--- Evaluating: {name} (Run {run_id}) ---")
        all_split_metrics[name] = {} # Initialize metrics dict for this model

        for split_name, loader in dataloaders.items():
            print(f"  Evaluating on: {split_name} set...")
            if loader is None:
                print(f"    Skipping {split_name} set (loader not available).")
                all_split_metrics[name][split_name] = {m: np.nan for m in cfg.METRICS}
                continue

            # Predict
            true_values_scaled, predictions_scaled = predict(model, loader, cfg.DEVICE)

            if true_values_scaled is None or predictions_scaled is None:
                 print(f"    Prediction failed for {name} on {split_name} set.")
                 all_split_metrics[name][split_name] = {m: np.nan for m in cfg.METRICS}
                 continue

            # Inverse transform
            n_samples, horizon, n_features = predictions_scaled.shape
            pred_reshaped = predictions_scaled.view(-1, n_features).cpu().numpy()
            true_reshaped = true_values_scaled.view(-1, n_features).cpu().numpy()

            try:
                if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                     raise ValueError("Scaler is not fitted.")
                predictions_original = scaler.inverse_transform(pred_reshaped)
                true_values_original = scaler.inverse_transform(true_reshaped)
            except Exception as e:
                print(f"    Error during inverse transform for {split_name}: {e}. Using scaled values.")
                predictions_original = pred_reshaped
                true_values_original = true_reshaped

            # Reshape back
            predictions_original = predictions_original.reshape(n_samples, horizon, n_features)
            true_values_original = true_values_original.reshape(n_samples, horizon, n_features)

            # Calculate metrics
            split_metrics = calculate_metrics(true_values_original, predictions_original)
            all_split_metrics[name][split_name] = split_metrics

            print(f"    Metrics ({split_name}): {split_metrics}")

            # Plotting predictions for the test set only for brevity
            if split_name == 'test':
                 plot_save_path = os.path.join(plots_run_dir, f"{name}_test_predictions.png")
                 utils.plot_predictions(true_values_original, predictions_original,
                                        title=f"{name} - Test Set Predictions (Run {run_id})",
                                        save_path=plot_save_path, series_idx=0, sample_idx_to_plot=0) # Explicitly plot the first sample


        # --- Robustness Evaluation (on Test Set only) ---
        if split_name == 'test' and cfg.ROBUSTNESS_NOISE_LEVELS and len(cfg.ROBUSTNESS_NOISE_LEVELS) > 0:
             print(f"\n--- Evaluating Robustness for {name}_run{run_id} (on Test Set) ---")
             df_robustness = evaluate_robustness(model, test_loader, cfg.DEVICE, scaler,
                                                 cfg.ROBUSTNESS_NOISE_LEVELS,
                                                 model_name=f"{name}_run{run_id}", metrics_dir=metrics_run_dir)
             all_robustness_dfs[name] = df_robustness


    # --- 8. 返回本次运行的结果 ---
    final_results = {
        'run_id': run_id,
        'seed': current_seed,
        'metrics': all_split_metrics, # Contains metrics for all models and splits
        'model_paths': model_paths
    }

    print(f"===== Finished Experiment Run {run_id + 1} / {cfg.STABILITY_RUNS} =====")
    return final_results


def update_config_from_args(cfg, args):
    """根据命令行参数更新配置对象"""
    if args.dataset_path:
        cfg.DATASET_PATH = args.dataset_path
        try:
            dataset_name_inferred = os.path.splitext(os.path.basename(cfg.DATASET_PATH))[0]
            if 'ETT-small' in cfg.DATASET_PATH:
                 parent_dir = os.path.basename(os.path.dirname(cfg.DATASET_PATH))
                 dataset_name_inferred = f"{parent_dir}_{dataset_name_inferred}"
            cfg.DATASET_NAME_FOR_RESULT_PATH = dataset_name_inferred
        except Exception:
            cfg.DATASET_NAME_FOR_RESULT_PATH = "unknown_dataset"
            print(f"Warning: Could not extract dataset name from path '{cfg.DATASET_PATH}'. Using default.")
    else:
        try: # Try inferring from default path if not provided
            dataset_name_inferred = os.path.splitext(os.path.basename(cfg.DATASET_PATH))[0] # Define inferred name here
            if 'ETT-small' in cfg.DATASET_PATH:
                 parent_dir = os.path.basename(os.path.dirname(cfg.DATASET_PATH))
                 dataset_name_inferred = f"{parent_dir}_{dataset_name_inferred}" # Use inferred name
            cfg.DATASET_NAME_FOR_RESULT_PATH = dataset_name_inferred # Assign inferred name
        except Exception:
            cfg.DATASET_NAME_FOR_RESULT_PATH = "unknown_dataset"


    if args.prediction_horizon is not None: cfg.PREDICTION_HORIZON = args.prediction_horizon
    if args.lookback_window is not None: cfg.LOOKBACK_WINDOW = args.lookback_window
    if args.epochs is not None: cfg.EPOCHS = args.epochs
    if args.stability_runs is not None: cfg.STABILITY_RUNS = args.stability_runs
    if args.teacher_model_name: cfg.TEACHER_MODEL_NAME = args.teacher_model_name if args.teacher_model_name.lower() != 'none' else None
    if args.student_model_name: cfg.STUDENT_MODEL_NAME = args.student_model_name

    # --- RDT Alpha Scheduler Arguments ---
    if args.alpha_schedule: cfg.ALPHA_SCHEDULE = args.alpha_schedule
    if args.alpha_start is not None: cfg.ALPHA_START = args.alpha_start
    if args.alpha_end is not None: cfg.ALPHA_END = args.alpha_end
    if args.constant_alpha is not None: cfg.CONSTANT_ALPHA = args.constant_alpha

    # --- Control Gate Scheduler Arguments ---
    if args.control_gate_metric: cfg.CONTROL_GATE_METRIC = args.control_gate_metric
    if args.control_gate_threshold_low is not None: cfg.CONTROL_GATE_THRESHOLD_LOW = args.control_gate_threshold_low
    if args.control_gate_threshold_high is not None: cfg.CONTROL_GATE_THRESHOLD_HIGH = args.control_gate_threshold_high
    if args.control_gate_alpha_adjust_rate is not None: cfg.CONTROL_GATE_ALPHA_ADJUST_RATE = args.control_gate_alpha_adjust_rate
    if args.control_gate_target_similarity is not None: cfg.CONTROL_GATE_TARGET_SIMILARITY = args.control_gate_target_similarity
    if args.control_gate_mse_student_target is not None: cfg.CONTROL_GATE_MSE_STUDENT_TARGET = args.control_gate_mse_student_target

    # --- Early Stopping Based Scheduler Arguments ---
    if args.es_alpha_patience is not None: cfg.ES_ALPHA_PATIENCE = args.es_alpha_patience
    if args.es_alpha_adjust_mode: cfg.ES_ALPHA_ADJUST_MODE = args.es_alpha_adjust_mode
    if args.es_alpha_adjust_rate is not None: cfg.ES_ALPHA_ADJUST_RATE = args.es_alpha_adjust_rate


    # Update dependent configs - Needs to be more robust if models change
    # This assumes TEACHER_CONFIG and STUDENT_CONFIG exist and have these keys
    if hasattr(cfg, 'TEACHER_CONFIG') and cfg.TEACHER_CONFIG:
        cfg.TEACHER_CONFIG['h'] = cfg.PREDICTION_HORIZON
        cfg.TEACHER_CONFIG['input_size'] = cfg.LOOKBACK_WINDOW
        if hasattr(cfg, 'TARGET_COLS'): cfg.TEACHER_CONFIG['n_series'] = len(cfg.TARGET_COLS) # Ensure n_series is updated too
    if hasattr(cfg, 'STUDENT_CONFIG') and cfg.STUDENT_CONFIG:
        cfg.STUDENT_CONFIG['h'] = cfg.PREDICTION_HORIZON
        cfg.STUDENT_CONFIG['input_size'] = cfg.LOOKBACK_WINDOW
        if hasattr(cfg, 'TARGET_COLS'): cfg.STUDENT_CONFIG['n_series'] = len(cfg.TARGET_COLS)

    # Update other known model configs if they exist
    model_configs_to_update = ['NLINEAR_CONFIG', 'LSTM_CONFIG', 'AUTOFORMER_CONFIG', 'PATCHTST_CONFIG', 'DLINEAR_CONFIG'] # Add others as needed
    if hasattr(cfg, 'TARGET_COLS'): # Only update if TARGET_COLS exists
        target_cols_len = len(cfg.TARGET_COLS)
        for config_name in model_configs_to_update:
            if hasattr(cfg, config_name):
                model_cfg = getattr(cfg, config_name)
                if model_cfg: # Check if it's not None
                    model_cfg['h'] = cfg.PREDICTION_HORIZON
                    model_cfg['n_series'] = target_cols_len
                    if 'input_size' in model_cfg: model_cfg['input_size'] = cfg.LOOKBACK_WINDOW
                    if 'lookback' in model_cfg: model_cfg['lookback'] = cfg.LOOKBACK_WINDOW
                    if 'output_size' in model_cfg: model_cfg['output_size'] = target_cols_len # For RNN/LSTM

    # Update experiment name
    teacher_name_part = cfg.TEACHER_MODEL_NAME if cfg.TEACHER_MODEL_NAME else "NoTeacher"
    cfg.EXPERIMENT_NAME = f"RDT_{cfg.STUDENT_MODEL_NAME}_vs_{teacher_name_part}_h{cfg.PREDICTION_HORIZON}"

    # Ensure required metrics are present
    required_metrics = ['mse', 'mae', 'mape', 'wape']
    if not hasattr(cfg, 'METRICS') or not isinstance(cfg.METRICS, list):
        cfg.METRICS = required_metrics
    else:
        # Make a copy to avoid modifying the original list during iteration if needed
        current_metrics = list(cfg.METRICS)
        for m in required_metrics:
            if m not in current_metrics:
                cfg.METRICS.append(m) # Append to the original cfg.METRICS

    return cfg

def main(args):
    """主函数，处理参数并运行实验"""
    cfg = default_config
    cfg = update_config_from_args(cfg, args)

    # 在配置更新后，立即设置日志
    setup_logging(cfg.LOG_FILE_PATH, cfg.LOG_LEVEL)
    logging.info("--- Updated Configuration ---")
    for key, value in vars(cfg).items():
        # Filter out built-ins, callables, modules for cleaner print
        if not key.startswith('__') and not callable(value) and not isinstance(value, type(os)) and key not in ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']:
            logging.info(f"{key}: {value}")
    logging.info("---------------------------")

    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = getattr(cfg, 'DATASET_NAME_FOR_RESULT_PATH', 'unknown_dataset')
    teacher_name_part = cfg.TEACHER_MODEL_NAME if cfg.TEACHER_MODEL_NAME else "NoTeacher"
    student_name_part = cfg.STUDENT_MODEL_NAME
    experiment_dir = os.path.join(
        cfg.RESULTS_DIR,
        f"{dataset_name_for_path}_{teacher_name_part}_{student_name_part}_h{cfg.PREDICTION_HORIZON}_{start_timestamp}"
    )
    metrics_dir = os.path.join(experiment_dir, 'metrics')
    models_dir = os.path.join(experiment_dir, 'models')
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"--- Experiment results will be saved to: {experiment_dir} ---")

    all_run_results = []
    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")
    cfg.DEVICE = str(device) # Ensure cfg reflects actual device used

    # --- Run stability experiments ---
    for i in range(cfg.STABILITY_RUNS):
        results = run_single_experiment(cfg, run_id=i, models_dir=models_dir, plots_dir=plots_dir, metrics_dir=metrics_dir)
        if results:
            all_run_results.append(results)
        else:
            logging.warning(f"Run {i+1} failed, skipping.")
        if cfg.DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # --- Process and save final results ---
    if not all_run_results:
        logging.info("\nNo experiments completed successfully.")
        return

    # --- Calculate and save average metrics ---
    logging.info("\n--- Calculating Average Metrics Across Runs ---")
    average_metrics_data = []
    # Use the keys from the first successful run's metrics dict to determine ran models
    if not all_run_results: # Check again, just in case
        logging.info("No successful runs to process for averaging.")
        return

    # Find the first successful run to get model keys
    first_successful_run = next((r for r in all_run_results if r and 'metrics' in r and r['metrics']), None)
    if not first_successful_run:
        logging.warning("Could not find any successful run with metrics to determine model types.")
        return

    ran_models = list(first_successful_run['metrics'].keys())
    splits = ['train', 'val', 'test']
    metrics_to_average = cfg.METRICS

    for model_type in ran_models:
        for split in splits:
            for metric_name in metrics_to_average:
                metric_values = []
                for run_result in all_run_results:
                    # Check if run_result and nested keys exist before accessing
                    if run_result and 'metrics' in run_result and \
                       model_type in run_result['metrics'] and \
                       split in run_result['metrics'][model_type] and \
                       metric_name in run_result['metrics'][model_type][split]:
                        value = run_result['metrics'][model_type][split][metric_name]
                        if value is not None and not np.isnan(value):
                            metric_values.append(value)
                    # else: # Log missing data less verbosely or handle as needed
                    #     pass

                if metric_values:
                    average_value = np.mean(metric_values)
                    average_metrics_data.append({
                        'split': split,
                        'model_type': model_type,
                        'metric': metric_name,
                        'value': average_value
                    })
                    logging.info(f"  Avg {model_type} - {split} - {metric_name.upper()}: {average_value:.6f}")
                else:
                    logging.info(f"  Avg {model_type} - {split} - {metric_name.upper()}: N/A (No valid data across runs)")
                    average_metrics_data.append({
                        'split': split,
                        'model_type': model_type,
                        'metric': metric_name,
                        'value': np.nan
                    })

    # Save average metrics
    if average_metrics_data:
        avg_metrics_df = pd.DataFrame(average_metrics_data)
        # Use the main start_timestamp for the average file name for consistency
        avg_metrics_save_path = os.path.join(metrics_dir, f"average_metrics_{start_timestamp}.csv")
        try:
            avg_metrics_df.to_csv(avg_metrics_save_path, index=False, float_format='%.6f')
            logging.info(f"\nAverage metrics saved to {avg_metrics_save_path}")
        except Exception as e:
            logging.error(f"\nError saving average metrics CSV: {e}")
    else:
        logging.info("\nNo average metrics calculated.")

    # --- Optional: Save detailed run results if needed ---
    # save_detailed_runs = False # Set to True to save detailed run info
    # if save_detailed_runs:
    #     try:
    #         import pickle
    #         detailed_pickle_path = os.path.join(metrics_dir, f"all_runs_detailed_{start_timestamp}.pkl")
    #         with open(detailed_pickle_path, 'wb') as f:
    #             pickle.dump(all_run_results, f)
    #         logging.info(f"Detailed run results saved to {detailed_pickle_path}")
    #     except Exception as e:
    #         logging.error(f"\nError saving detailed run results: {e}")


    logging.info("\n--- Experiment Finished ---")
    logging.info(f"Find results (models, plots, average metrics) in: {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Time Series Forecasting Experiments with Optional RDT")
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset CSV file.')
    parser.add_argument('--prediction_horizon', type=int, help=f'Prediction horizon (default: {default_config.PREDICTION_HORIZON})')
    parser.add_argument('--lookback_window', type=int, help=f'Lookback window size (default: {default_config.LOOKBACK_WINDOW})')
    parser.add_argument('--epochs', type=int, help=f'Number of training epochs (default: {default_config.EPOCHS})')
    parser.add_argument('--stability_runs', type=int, default=default_config.STABILITY_RUNS, help=f'Number of runs for stability analysis (default: {default_config.STABILITY_RUNS})')
    parser.add_argument('--teacher_model_name', type=str, default=default_config.TEACHER_MODEL_NAME,
                        help=f'Name of the teacher model (e.g., DLinear, PatchTST, None) (default: {default_config.TEACHER_MODEL_NAME})')
    parser.add_argument('--student_model_name', type=str, default=default_config.STUDENT_MODEL_NAME,
                        help=f'Name of the student model (e.g., PatchTST, DLinear) (default: {default_config.STUDENT_MODEL_NAME})')

    # --- RDT Alpha Scheduler Arguments ---
    parser.add_argument('--alpha_schedule', type=str, default=default_config.ALPHA_SCHEDULE,
                        help=f'Alpha schedule type (linear, cosine, fixed, early_stopping_based, control_gate) (default: {default_config.ALPHA_SCHEDULE})')
    parser.add_argument('--alpha_start', type=float, default=default_config.ALPHA_START,
                        help=f'Starting alpha value for linear/cosine schedules (default: {default_config.ALPHA_START})')
    parser.add_argument('--alpha_end', type=float, default=default_config.ALPHA_END,
                        help=f'Ending alpha value for linear/cosine schedules (default: {default_config.ALPHA_END})')
    parser.add_argument('--constant_alpha', type=float, default=default_config.CONSTANT_ALPHA,
                        help=f'Constant alpha value for fixed schedule (default: {default_config.CONSTANT_ALPHA})')

    # --- Control Gate Scheduler Arguments ---
    parser.add_argument('--control_gate_metric', type=str, default=default_config.CONTROL_GATE_METRIC,
                        help=f'Metric for control gate (cosine_similarity, mse_student_true, mse_student_teacher) (default: {default_config.CONTROL_GATE_METRIC})')
    parser.add_argument('--control_gate_threshold_low', type=float, default=default_config.CONTROL_GATE_THRESHOLD_LOW,
                        help=f'Lower threshold for control gate (default: {default_config.CONTROL_GATE_THRESHOLD_LOW})')
    parser.add_argument('--control_gate_threshold_high', type=float, default=default_config.CONTROL_GATE_THRESHOLD_HIGH,
                        help=f'Higher threshold for control gate (default: {default_config.CONTROL_GATE_THRESHOLD_HIGH})')
    parser.add_argument('--control_gate_alpha_adjust_rate', type=float, default=default_config.CONTROL_GATE_ALPHA_ADJUST_RATE,
                        help=f'Alpha adjustment rate for control gate (default: {default_config.CONTROL_GATE_ALPHA_ADJUST_RATE})')
    parser.add_argument('--control_gate_target_similarity', type=float, default=default_config.CONTROL_GATE_TARGET_SIMILARITY,
                        help=f'Target similarity for control gate (optional) (default: {default_config.CONTROL_GATE_TARGET_SIMILARITY})')
    parser.add_argument('--control_gate_mse_student_target', type=float, default=default_config.CONTROL_GATE_MSE_STUDENT_TARGET,
                        help=f'Target MSE for student vs true for control gate (optional) (default: {default_config.CONTROL_GATE_MSE_STUDENT_TARGET})')

    # --- Early Stopping Based Scheduler Arguments ---
    parser.add_argument('--es_alpha_patience', type=int, default=default_config.ES_ALPHA_PATIENCE,
                        help=f'Patience for early stopping based alpha scheduler (default: {default_config.ES_ALPHA_PATIENCE})')
    parser.add_argument('--es_alpha_adjust_mode', type=str, default=default_config.ES_ALPHA_ADJUST_MODE,
                        help=f'Alpha adjustment mode for early stopping (freeze, decay_to_teacher, decay_to_student) (default: {default_config.ES_ALPHA_ADJUST_MODE})')
    parser.add_argument('--es_alpha_adjust_rate', type=float, default=default_config.ES_ALPHA_ADJUST_RATE,
                        help=f'Alpha adjustment rate for early stopping decay modes (default: {default_config.ES_ALPHA_ADJUST_RATE})')

    args = parser.parse_args()
    main(args)
