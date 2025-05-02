import torch
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# --- 导入自定义模块 ---
from src import config as default_config
from src import utils # 确保 utils.py 中包含新的绘图函数
from src.data_handler import load_and_preprocess_data
from src.models import get_teacher_model, get_student_model
from src.trainer import StandardTrainer, RDT_Trainer, get_optimizer, get_loss_function
from src.schedulers import get_alpha_scheduler, ConstantScheduler # 明确导入 ConstantScheduler
from src.evaluator import evaluate_model, evaluate_robustness, calculate_metrics


def run_single_experiment(cfg, run_id=0):
    """运行单次完整的实验流程（用于稳定性评估）"""
    print(f"\n===== Starting Experiment Run {run_id + 1} / {cfg.STABILITY_RUNS} with Seed {cfg.SEED + run_id} =====")
    current_seed = cfg.SEED + run_id
    utils.set_seed(current_seed)

    run_results = {'run_id': run_id, 'seed': current_seed}
    model_paths = {} # 存储本次运行的模型路径

    # --- 1. 数据加载与预处理 ---
    try:
        train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(cfg)
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return None # 无法继续

    # --- 2. 初始化模型、优化器、损失函数 ---
    teacher_model = get_teacher_model(cfg)
    student_model_base = get_student_model(cfg) # 用于标准训练和 RDT 的基础学生模型

    task_loss_fn = get_loss_function(cfg)
    distill_loss_fn = get_loss_function(cfg) # RDT 中蒸馏损失通常与任务损失一致

    # --- 3. 训练教师模型 (Standard Training) ---
    print("\n--- Training Teacher Model ---")
    teacher_optimizer = get_optimizer(teacher_model, cfg)
    teacher_model_save_path = os.path.join(cfg.MODELS_DIR, f"teacher_{cfg.TEACHER_MODEL_NAME}_run{run_id}_seed{current_seed}.pt")
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
    utils.plot_losses(teacher_history['train_loss'], teacher_history['val_loss'],
                      title=f"Teacher ({cfg.TEACHER_MODEL_NAME}) Training Loss (Run {run_id})",
                      save_path=os.path.join(cfg.PLOTS_DIR, f"teacher_loss_run{run_id}.png"))

    # --- 4. 训练基线学生模型 (Standard Training, Alpha=1.0 Task Only) ---
    print("\n--- Training Baseline Student Model (Task Only, Alpha=1.0) ---")
    student_task_only_model = get_student_model(cfg)
    student_task_only_optimizer = get_optimizer(student_task_only_model, cfg)
    student_task_only_save_path = os.path.join(cfg.MODELS_DIR, f"student_{cfg.STUDENT_MODEL_NAME}_task_only_run{run_id}_seed{current_seed}.pt")
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
        model_name=f"Student ({cfg.STUDENT_MODEL_NAME}) Task Only (Alpha=1.0)"
    )
    trained_student_task_only_model, student_task_only_history = student_task_only_trainer.train()
    utils.plot_losses(student_task_only_history['train_loss'], student_task_only_history['val_loss'],
                      title=f"Student Task Only (Alpha=1.0) Training Loss (Run {run_id})",
                      save_path=os.path.join(cfg.PLOTS_DIR, f"student_task_only_loss_run{run_id}.png"))

    # --- 5. 训练 RDT 学生模型 ---
    print("\n--- Training RDT Student Model ---")
    student_rdt_model = get_student_model(cfg)
    student_rdt_optimizer = get_optimizer(student_rdt_model, cfg)
    alpha_scheduler = get_alpha_scheduler(cfg) # RDT 特有
    student_rdt_save_path = os.path.join(cfg.MODELS_DIR, f"student_{cfg.STUDENT_MODEL_NAME}_rdt_run{run_id}_seed{current_seed}.pt")
    model_paths['student_rdt'] = student_rdt_save_path
    rdt_trainer = RDT_Trainer(
        student_model=student_rdt_model,
        teacher_model=trained_teacher_model, # 使用训练好的教师模型
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
    utils.plot_losses(student_rdt_history['train_loss'], student_rdt_history['val_loss'],
                      title=f"RDT Student Training Loss (Total Train, Task Val) (Run {run_id})",
                      save_path=os.path.join(cfg.PLOTS_DIR, f"student_rdt_loss_run{run_id}.png"))
    # (可选) 绘制 Alpha 变化图
    if 'alpha' in student_rdt_history and student_rdt_history['alpha']:
        plt.figure()
        plt.plot(student_rdt_history['alpha'])
        plt.title(f"RDT Alpha Schedule (Run {run_id})")
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.grid(True)
        plt.savefig(os.path.join(cfg.PLOTS_DIR, f"student_rdt_alpha_run{run_id}.png"))
        plt.close()

    # --- 6. 训练追随者学生模型 (Follower, Alpha=0.0) ---
    print("\n--- Training Follower Student Model (Distill Only, Alpha=0.0) ---")
    student_follower_model = get_student_model(cfg)
    student_follower_optimizer = get_optimizer(student_follower_model, cfg)
    follower_alpha_scheduler = ConstantScheduler(alpha_value=0.0, total_epochs=cfg.EPOCHS)
    student_follower_save_path = os.path.join(cfg.MODELS_DIR, f"student_{cfg.STUDENT_MODEL_NAME}_follower_run{run_id}_seed{current_seed}.pt")
    model_paths['student_follower'] = student_follower_save_path
    follower_trainer = RDT_Trainer( # 复用 RDT Trainer，但 Alpha 恒为 0
        student_model=student_follower_model,
        teacher_model=trained_teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=student_follower_optimizer,
        task_loss_fn=task_loss_fn,
        distill_loss_fn=distill_loss_fn,
        alpha_scheduler=follower_alpha_scheduler, # 使用 alpha=0 的调度器
        device=cfg.DEVICE,
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE, # 注意：早停基于 Task Loss，即使训练目标是 Distill Loss
        model_save_path=student_follower_save_path,
        model_name=f"Student ({cfg.STUDENT_MODEL_NAME}) Follower (Alpha=0.0)"
    )
    trained_student_follower_model, student_follower_history = follower_trainer.train()
    utils.plot_losses(student_follower_history['train_loss'], student_follower_history['val_loss'],
                       title=f"Follower Student Training Loss (Distill Train, Task Val) (Run {run_id})",
                       save_path=os.path.join(cfg.PLOTS_DIR, f"student_follower_loss_run{run_id}.png"))

    # --- 7. 评估所有模型 ---
    models_to_evaluate = {
        "Teacher": trained_teacher_model,
        "Student_TaskOnly": trained_student_task_only_model,
        "Student_RDT": trained_student_rdt_model,
        "Student_Follower": trained_student_follower_model
    }
    all_metrics = {}
    all_predictions = {}
    all_robustness_dfs = {} # <<< 新增: 收集鲁棒性结果
    true_values_original = None

    print("\n--- Evaluating Models on Test Set ---")
    for name, model in models_to_evaluate.items():
        print(f"Evaluating {name}...")
        model_display_name = f"{name}_run{run_id}" # 为本次运行的模型添加标识
        metrics, trues, preds = evaluate_model(model, test_loader, cfg.DEVICE, scaler, model_name=model_display_name)
        all_metrics[name] = metrics # 存储本次运行的核心指标
        all_predictions[name] = preds # 存储预测结果
        if true_values_original is None:
            true_values_original = trues # 只存储一次真实值

        # --- (可选) 评估鲁棒性 ---
        if cfg.ROBUSTNESS_NOISE_LEVELS and len(cfg.ROBUSTNESS_NOISE_LEVELS) > 0:
             print(f"--- Evaluating Robustness for {name}_run{run_id} ---")
             # <<< 修改: 接收并存储鲁棒性评估结果 DataFrame >>>
             df_robustness = evaluate_robustness(model, test_loader, cfg.DEVICE, scaler,
                                                 cfg.ROBUSTNESS_NOISE_LEVELS, model_name=f"{name}_run{run_id}")
             all_robustness_dfs[name] = df_robustness # <<< 存储DataFrame
             # <<< ---------------------------------------- >>>

    # --- 绘制对比图 ---
    print("\n--- Generating Comparison Plots ---")
    # 1. 预测结果对比图
    if true_values_original is not None and all_predictions:
        comp_plot_save_path = os.path.join(cfg.PLOTS_DIR, f"comparison_predictions_run{run_id}.png")
        utils.plot_comparison_predictions(
            true_values_original,
            all_predictions, # 传递包含所有模型预测的字典
            title=f"Test Set Prediction Comparison (Run {run_id})",
            save_path=comp_plot_save_path,
            series_idx=0 # 可以按需修改要绘制的序列索引
        )

    # 2. 性能指标对比图
    if all_metrics:
        metric_comp_plot_save_path = os.path.join(cfg.PLOTS_DIR, f"comparison_metrics_run{run_id}.png")
        # <<< 调用 plot_metric_comparison >>>
        utils.plot_metric_comparison(
            all_metrics, # 传递包含所有模型指标的字典
            title=f"Test Set Metric Comparison (Run {run_id})",
            save_path=metric_comp_plot_save_path
        )
        # <<< -------------------------- >>>

    # 3. 鲁棒性对比图
    if all_robustness_dfs and cfg.METRICS:
        for metric in cfg.METRICS: # 为配置中的每个指标绘制鲁棒性图
             # <<< 调用 plot_robustness_comparison >>>
             robustness_comp_plot_save_path = os.path.join(cfg.PLOTS_DIR, f"comparison_robustness_{metric}_run{run_id}.png")
             utils.plot_robustness_comparison(
                 all_robustness_dfs, # 传递包含所有模型鲁棒性结果的字典
                 metric_name=metric,
                 title=f"Robustness Comparison ({metric.upper()}) vs Noise Level (Run {run_id})",
                 save_path=robustness_comp_plot_save_path
                 # Optional: pass all_metrics here if you want to add noise=0 point
             )
             # <<< ------------------------------- >>>
    # ------------------------

    # --- 8. 收集本次运行的结果 ---
    for model_name, metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            run_results[f"{model_name}_{metric_name}"] = value
    # (可选) 将鲁棒性指标也添加到结果中，但会使 DataFrame 列变多
    # for model_name, df_robust in all_robustness_dfs.items():
    #     for noise_level in df_robust.index:
    #         for metric_name in df_robust.columns:
    #             run_results[f"{model_name}_robust_{noise_level}_{metric_name}"] = df_robust.loc[noise_level, metric_name]

    run_results['teacher_model_path'] = model_paths.get('teacher', 'N/A')
    run_results['student_task_only_model_path'] = model_paths.get('student_task_only', 'N/A')
    run_results['student_rdt_model_path'] = model_paths.get('student_rdt', 'N/A')
    run_results['student_follower_model_path'] = model_paths.get('student_follower', 'N/A')

    print(f"===== Finished Experiment Run {run_id + 1} / {cfg.STABILITY_RUNS} =====")
    return run_results


def main(args):
    """主函数，处理参数并运行实验"""
    # --- 加载和合并配置 ---
    # (如果使用 argparse, 在这里根据 args 更新 cfg)
    # e.g., default_config.EPOCHS = args.epochs if args.epochs is not None else default_config.EPOCHS
    cfg = default_config
    print("--- Current Configuration ---")
    # 打印配置信息 (过滤内置属性和模块对象)
    for key, value in vars(cfg).items():
        if not key.startswith('__') and not callable(value) and not isinstance(value, type(os)):
            print(f"{key}: {value}")
    print("---------------------------")

    # --- <<< 创建动态实验结果目录 >>> ---
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # 尝试从完整路径中提取数据集文件名（不含扩展名）
        dataset_name = os.path.splitext(os.path.basename(cfg.DATASET_PATH))[0]
    except Exception:
        dataset_name = "unknown_dataset" # 如果提取失败，使用默认名称
        print(f"Warning: Could not extract dataset name from path '{cfg.DATASET_PATH}'. Using '{dataset_name}'.")
    experiment_dir = os.path.join(
        cfg.RESULTS_DIR, # 使用 config.py 中定义的基础结果目录
        f"{dataset_name}_{cfg.TEACHER_MODEL_NAME}_{cfg.STUDENT_MODEL_NAME}_{start_timestamp}"
    )
    # 创建实验主目录和子目录
    metrics_dir = os.path.join(experiment_dir, 'metrics')
    models_dir = os.path.join(experiment_dir, 'models')
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"--- Experiment results will be saved to: {experiment_dir} ---")

    all_run_results = []
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_name = f"{cfg.EXPERIMENT_NAME}_{start_timestamp}.csv"
    main_results_filename = f"all_runs_summary_{start_timestamp}.csv" # 主结果文件名
    
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"当前设备: {device}")
        print(f"使用的 CUDA 设备编号: {torch.cuda.current_device()}")
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"当前设备: {device}")

    # --- 运行多次实验以评估稳定性 ---
    for i in range(cfg.STABILITY_RUNS):
        results = run_single_experiment(cfg, models_dir, plots_dir, metrics_dir, run_id=i)
        if results:
            all_run_results.append(results)
        else:
            print(f"Run {i+1} failed, skipping.")
        # 强制清理 GPU 缓存 (可能有助于长时间运行)
        if cfg.DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # --- 处理和保存最终结果 ---
    if not all_run_results:
        print("\nNo experiments completed successfully.")
        return

    results_df = pd.DataFrame(all_run_results)

    # --- 计算稳定性指标 (均值和标准差) ---
    stability_summary = None
    if cfg.STABILITY_RUNS > 1:
        print("\n--- Stability Analysis (Across Runs) ---")
        # <<< 优化: 基于 cfg.METRICS 选择列 >>>
        metric_cols_to_summarize = []
        model_prefixes = ["Teacher", "Student_TaskOnly", "Student_RDT", "Student_Follower"] # 确保与 run_single_experiment 中一致
        for prefix in model_prefixes:
            for metric in cfg.METRICS:
                col_name = f"{prefix}_{metric}"
                if col_name in results_df.columns:
                    metric_cols_to_summarize.append(col_name)
        # <<< ---------------------------- >>>

        if metric_cols_to_summarize:
             stability_summary = results_df[metric_cols_to_summarize].agg(['mean', 'std'])
             print("--- Stability Summary (Mean & Std) ---")
             print(stability_summary)
             summary_save_path = os.path.join(metrics_dir, f"stability_summary_{start_timestamp}.csv")
             stability_summary.to_csv(summary_save_path)
             print(f"Stability summary saved to {summary_save_path}")

             # --- 绘制稳定性对比图 ---
             if cfg.METRICS:
                 print("\n--- Generating Stability Comparison Plots ---")
                 for metric in cfg.METRICS:
                     # <<< 调用 plot_stability_comparison >>>
                     stability_plot_save_path = os.path.join(plots_dir, f"comparison_stability_{metric}_{start_timestamp}.png")
                     utils.plot_stability_comparison(
                         results_df,
                         metric_to_plot=metric, # 指标名称 (mae, mse)
                         title=f"Stability Comparison ({metric.upper()}) Across {cfg.STABILITY_RUNS} Runs",
                         save_path=stability_plot_save_path,
                         plot_type='box' # 或 'violin'
                     )
                     # <<< --------------------------------- >>>
             # ----------------------------
        else:
             print("No metric columns found for stability summary based on config.")


    # --- 保存所有运行的详细结果 ---
    all_results_save_path = os.path.join(metrics_dir, main_results_filename)
    results_df.to_csv(all_results_save_path, index=False)
    print(f"\nAll run results saved to {all_results_save_path}")

    print("\n--- Experiment Finished ---")
    print(f"Find models in: {cfg.MODELS_DIR}")
    print(f"Find plots in: {cfg.PLOTS_DIR}")
    print(f"Find metrics in: {cfg.METRICS_DIR}")


if __name__ == "__main__":
    # --- 参数解析 (保持简单，主要使用 config.py) ---
    parser = argparse.ArgumentParser(description="RDT Framework for Time Series Forecasting")
    # 可以在这里添加少量关键参数覆盖配置，例如
    # parser.add_argument('--epochs', type=int, help=f'Override number of training epochs (default: {default_config.EPOCHS})')
    # parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help=f'Override device (default: {default_config.DEVICE})')
    args = parser.parse_args()

    # --- (可选) 更新配置 ---
    # if args.epochs is not None: default_config.EPOCHS = args.epochs
    # if args.device is not None: default_config.DEVICE = args.device
    # # 确保使用最新的 DEVICE 值
    # if default_config.DEVICE == 'cuda' and not torch.cuda.is_available():
    #     print("Warning: CUDA requested but not available, falling back to CPU.")
    #     default_config.DEVICE = 'cpu'
    # elif default_config.DEVICE == 'cpu':
    #      if torch.cuda.is_available():
    #          print("Info: CPU selected, but CUDA is available. Consider using '--device cuda'.")

    # --- 运行主程序 ---
    main(args) # 传递解析后的参数（即使为空）
