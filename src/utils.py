import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from src import config
import seaborn as sns 
import logging
import sys

# --- 日志配置函数 ---
def setup_logging(log_file_path, log_level_str='INFO'):
    """
    设置全局日志记录器，将日志同时输出到控制台和文件。
    Args:
        log_file_path (str): 日志文件的完整路径。
        log_level_str (str): 日志级别字符串 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')。
    """
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level_str.upper(), logging.INFO)) # 根据字符串设置级别

    # 清除已有的处理器，防止重复输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    logging.info(f"Logging configured. Log file: {log_file_path}, Level: {log_level_str}")
    return logger

# plt.style.use('seaborn-v0_8-grid') # 使用 seaborn 风格 (注释掉或删除原行)
plt.style.use('seaborn-v0_8-whitegrid') # 替换为这个常用的 seaborn 风格

def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    # 这些设置可以提高确定性，但可能会牺牲性能
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def plot_losses(train_losses, val_losses, title, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")

def plot_training_metrics(history, save_dir, model_name="Model"):
    """
    绘制训练过程中的各项指标曲线，包括训练损失、验证损失、学习率、梯度范数和评估指标。
    Args:
        history (dict): 包含训练历史的字典，例如 {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': [], 'val_rmse': [], ...}
        save_dir (str): 图表保存的目录。
        model_name (str): 模型名称，用于图表标题。
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. 训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, f'{model_name.lower()}_loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss curve plot saved to {loss_plot_path}")

    # 2. 学习率曲线
    if 'lr' in history and history['lr']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['lr'], label='Learning Rate', color='green')
        plt.title(f'{model_name} Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.tight_layout()
        lr_plot_path = os.path.join(save_dir, f'{model_name.lower()}_lr_curve.png')
        plt.savefig(lr_plot_path)
        plt.close()
        logging.info(f"Learning rate curve plot saved to {lr_plot_path}")

    # 3. 梯度范数曲线
    if 'grad_norm' in history and history['grad_norm']:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['grad_norm'], label='Gradient Norm', color='purple')
        plt.title(f'{model_name} Gradient Norm per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.tight_layout()
        grad_norm_plot_path = os.path.join(save_dir, f'{model_name.lower()}_grad_norm_curve.png')
        plt.savefig(grad_norm_plot_path)
        plt.close()
        logging.info(f"Gradient norm curve plot saved to {grad_norm_plot_path}")

    # 4. 评估指标曲线 (遍历所有以 'val_' 开头的指标)
    val_metrics_to_plot = {k: v for k, v in history.items() if k.startswith('val_') and k != 'val_loss'}
    if val_metrics_to_plot:
        num_metrics = len(val_metrics_to_plot)
        # 动态调整子图布局，最多每行3个
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() # 将 axes 展平，方便迭代

        for i, (metric_name, metric_values) in enumerate(val_metrics_to_plot.items()):
            ax = axes[i]
            ax.plot(epochs, metric_values, label=metric_name.replace('val_', '').upper(), color='orange')
            ax.set_title(f'{model_name} Validation {metric_name.replace("val_", "").upper()}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric_name.replace('val_', '').upper())
            ax.legend()
            ax.grid(True)
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        metrics_plot_path = os.path.join(save_dir, f'{model_name.lower()}_val_metrics_curves.png')
        plt.savefig(metrics_plot_path)
        plt.close()
        logging.info(f"Validation metrics curves plot saved to {metrics_plot_path}")

def plot_predictions(y_true, y_pred, title, save_path, series_idx=0, sample_idx_to_plot=0, target_cols_list=None):
    """
    绘制单个样本的真实值与预测值的对比图 (绘制完整的预测范围)。
    Args:
        y_true (np.ndarray): 真实值数组。
        y_pred (np.ndarray): 预测值数组。
        title (str): 图表主标题。
        save_path (str): 图表保存路径。
        series_idx (int): 要绘制的序列索引。
        sample_idx_to_plot (int): 要绘制的样本索引。
        target_cols_list (list, optional): 目标列名称列表。默认为 None。
    """
    if y_true.ndim != 3 or y_pred.ndim != 3:
        print(f"Warning: Unexpected data dimensions for plotting. True: {y_true.ndim}, Pred: {y_pred.ndim}. Expected 3D [samples, horizon, features].")
        return
    if y_true.shape[0] <= sample_idx_to_plot or y_pred.shape[0] <= sample_idx_to_plot:
        print(f"Warning: sample_idx_to_plot ({sample_idx_to_plot}) is out of bounds for available samples ({y_true.shape[0]}). Plotting first sample.")
        sample_idx_to_plot = 0
    if y_true.shape[2] <= series_idx or y_pred.shape[2] <= series_idx:
        print(f"Warning: series_idx ({series_idx}) is out of bounds for available features ({y_true.shape[2]}). Plotting first feature.")
        series_idx = 0
    if y_true.shape != y_pred.shape:
         print(f"Warning: Shape mismatch between true values {y_true.shape} and predictions {y_pred.shape}. Plotting may be incorrect.")
         # Attempt to plot based on true shape's horizon if possible
         # return # Or decide how to handle, maybe return

    # 选择指定样本和序列的完整预测范围
    try:
        y_true_sample = y_true[sample_idx_to_plot, :, series_idx]
        y_pred_sample = y_pred[sample_idx_to_plot, :, series_idx]
        horizon = len(y_true_sample) # Get horizon length from the sample
    except IndexError as e:
        print(f"Error accessing data with sample_idx={sample_idx_to_plot}, series_idx={series_idx}: {e}")
        return

    plt.figure(figsize=(15, 7))
    time_steps = np.arange(horizon) # X-axis represents steps within the horizon

    plt.plot(time_steps, y_true_sample, label='True Values', marker='.', linestyle='-')
    plt.plot(time_steps, y_pred_sample, label='Predictions', marker='x', linestyle='--')

    # 如果提供了 TARGET_COLS，使用具体的列名
    if target_cols_list and series_idx < len(target_cols_list):
        target_col_name = target_cols_list[series_idx]
    else:
        target_col_name = f"Series {series_idx}"
    full_title = f"{title} - Sample {sample_idx_to_plot} - {target_col_name}" # Add sample index to title

    plt.title(full_title)
    plt.xlabel(f'Time Steps within Prediction Horizon (Length: {horizon})') # Update x-axis label
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot for sample {sample_idx_to_plot} saved to {save_path}")

def plot_residuals_analysis(y_true, y_pred, save_dir, model_name="Model", series_idx=0, target_cols_list=None):
    """
    对预测残差进行分析并绘制图表：残差时间序列图和残差分布直方图。
    Args:
        y_true (np.ndarray): 真实值数组 [num_samples, horizon, features]。
        y_pred (np.ndarray): 预测值数组 [num_samples, horizon, features]。
        save_dir (str): 图表保存的目录。
        model_name (str): 模型名称，用于图表标题。
        series_idx (int): 要分析的目标序列的索引。
        target_cols_list (list): 目标列名称列表。
    """
    os.makedirs(save_dir, exist_ok=True)

    if y_true.ndim != 3 or y_pred.ndim != 3:
        logging.warning(f"Unexpected data dimensions for residuals analysis. True: {y_true.ndim}, Pred: {y_pred.ndim}. Expected 3D [samples, horizon, features]. Skipping.")
        return
    if y_true.shape != y_pred.shape:
        logging.warning(f"Shape mismatch between true values {y_true.shape} and predictions {y_pred.shape}. Cannot perform residuals analysis. Skipping.")
        return
    if y_true.shape[2] <= series_idx:
        logging.warning(f"Series index {series_idx} is out of bounds for available features ({y_true.shape[2]}). Skipping residuals analysis.")
        return

    # # 提取指定序列的数据并展平
    # y_true_flat = y_true[:, :, series_idx].reshape(-1)
    # y_pred_flat = y_pred[:, :, series_idx].reshape(-1)
    # residuals = y_true_flat - y_pred_flat

    # if target_cols_list and series_idx < len(target_cols_list):
    #     target_col_name = target_cols_list[series_idx]
    # else:
    #     target_col_name = f"Series {series_idx}"
    
    # # 1. 残差时间序列图
    # plt.figure(figsize=(15, 6))
    # plt.plot(residuals, label='Residuals', color='red', alpha=0.7)
    # plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    # plt.title(f'{model_name} Residuals Time Series - {target_col_name}')
    # plt.xlabel('Time Point (Flattened)')
    # plt.ylabel('Residual Value')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # residuals_ts_path = os.path.join(save_dir, f'{model_name.lower()}_{target_col_name.lower().replace(" ", "_")}_residuals_ts.png')
    # plt.savefig(residuals_ts_path)
    # plt.close()
    # logging.info(f"Residuals time series plot saved to {residuals_ts_path}")

    # 2. 残差分布直方图 (默认不绘制)
    # plt.figure(figsize=(10, 6))
    # sns.histplot(residuals, kde=True, bins=50, color='skyblue')
    # plt.title(f'{model_name} Residuals Distribution - {target_col_name}')
    # plt.xlabel('Residual Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.tight_layout()
    # residuals_dist_path = os.path.join(save_dir, f'{model_name.lower()}_{target_col_name.lower().replace(" ", "_")}_residuals_distribution.png')
    # plt.savefig(residuals_dist_path)
    # plt.close()
    # logging.info(f"Residuals distribution plot saved to {residuals_dist_path}")

def save_results(metrics_dict, filename):
    """将指标字典保存为 CSV 文件"""
    df = pd.DataFrame([metrics_dict])
    filepath = os.path.join(config.METRICS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")

def load_model(model, model_path, device):
    """加载模型状态字典"""
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def save_model(model, model_path):
    """保存模型状态字典"""
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")

def plot_comparison_predictions(true_values, predictions_dict, title, save_path, series_idx=0, max_points=500, target_cols_list=None):
    """
    绘制真实值与多个模型预测值的对比图。
    Args:
        true_values (np.ndarray): 真实值数组 [num_samples, horizon, features]。
        predictions_dict (dict): 字典，键是模型名称 (str)，值是对应的预测值数组 (np.ndarray)。
        title (str): 图表标题。
        save_path (str): 图表保存路径。
        series_idx (int): 要绘制的目标序列的索引。
        max_points (int): 最多绘制的点数，防止图像过于拥挤。
    """
    if true_values.ndim != 3:
        print(f"Warning: True values have unexpected dimensions for plotting: {true_values.ndim}")
        return
    if not predictions_dict:
        print("Warning: No predictions provided for comparison plotting.")
        return
    plt.figure(figsize=(18, 8)) # 增加图像宽度以容纳更多线条
    # 提取指定序列的数据并展平
    try:
        y_true_flat = true_values[:, :, series_idx].reshape(-1)
    except IndexError:
        print(f"Error: series_idx {series_idx} is out of bounds for true values features ({true_values.shape[-1]})")
        return
    plot_len = min(max_points, len(y_true_flat))
    # 绘制真实值
    plt.plot(y_true_flat[:plot_len], label='True Values', color='black', linewidth=2, marker='.', linestyle='-')
    # 绘制每个模型的预测值
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict))) # 获取一组不同的颜色
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        if y_pred.ndim != 3:
             print(f"Warning: Predictions for model '{model_name}' have unexpected dimensions: {y_pred.ndim}")
             continue
        try:
            y_pred_flat = y_pred[:, :, series_idx].reshape(-1)
            if len(y_pred_flat) != len(y_true_flat):
                 print(f"Warning: Length mismatch between true values ({len(y_true_flat)}) and predictions for '{model_name}' ({len(y_pred_flat)})")
                 continue
            plt.plot(y_pred_flat[:plot_len], label=f'{model_name} Preds', color=colors[i], linestyle='--', alpha=0.8)
        except IndexError:
            print(f"Error: series_idx {series_idx} is out of bounds for '{model_name}' predictions features ({y_pred.shape[-1]})")
        except Exception as e:
             print(f"Error plotting predictions for {model_name}: {e}")
    if target_cols_list and series_idx < len(target_cols_list):
        target_col_name = target_cols_list[series_idx]
    else:
        target_col_name = f"Series {series_idx}"
    full_title = f"{title} - {target_col_name}"
    plt.title(full_title, fontsize=16)
    plt.xlabel('Time Steps (Sample Index within Test Set)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison prediction plot saved to {save_path}")

def plot_metric_comparison(metrics_dict, title, save_path):
    """
    使用条形图比较不同模型在测试集上的性能指标。
    Args:
        metrics_dict (dict): 字典，键是模型名称 (str)，值是包含指标名称和值的字典。
                             Example: {'Teacher': {'mae': 0.1, 'mse': 0.02}, 'Student_RDT': {...}}
        title (str): 图表标题。
        save_path (str): 图表保存路径。
    """
    if not metrics_dict:
        print("Warning: No metrics provided for comparison plotting.")
        return
    # 将字典转换为 DataFrame 以方便绘图
    df = pd.DataFrame(metrics_dict).T # 转置使得模型名为行索引
    if df.empty:
        print("Warning: Metrics DataFrame is empty.")
        return
    num_models = len(df.index)
    num_metrics = len(df.columns)
    bar_width = 0.8 / num_metrics # 调整宽度以适应指标数量
    index = np.arange(num_models)
    fig, ax = plt.subplots(figsize=(max(10, num_models * 1.5), 6)) # 动态调整宽度
    for i, metric_name in enumerate(df.columns):
        # 计算每个指标的条形位置
        bar_positions = index + i * bar_width - (bar_width * (num_metrics - 1) / 2)
        bars = ax.bar(bar_positions, df[metric_name], bar_width, label=metric_name.upper())
        # 在条形上方显示数值
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_ylabel('Metric Value (Lower is Better)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=10) # 旋转标签防止重叠
    ax.legend(title="Metrics", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metric comparison plot saved to {save_path}")

def plot_robustness_comparison(robustness_results_dict, metric_name, title, save_path):
    """
    绘制不同模型在不同噪声水平下指定性能指标的变化曲线。
    Args:
        robustness_results_dict (dict): 字典，键是模型名称(str)，值是 evaluate_robustness 返回的 DataFrame。
        metric_name (str): 要绘制的指标名称 (例如 'mae', 'mse')。
        title (str): 图表标题。
        save_path (str): 图表保存路径。
    """
    if not robustness_results_dict:
        print("Warning: No robustness results provided for comparison plotting.")
        return
    plt.figure(figsize=(12, 7))
    sns.set_palette("tab10") # 设置调色板
    # 添加 0 噪声水平下的原始性能，需要从 metrics_dict 获取
    # (假设 metrics_dict 在调用此函数前可用)
    # all_dfs = [] # 暂不需要
    for model_name, df_robustness in robustness_results_dict.items():
        if metric_name not in df_robustness.columns:
            print(f"Warning: Metric '{metric_name}' not found in robustness results for model '{model_name}'. Skipping.")
            continue
        # 提取噪声水平和对应的指标值
        # 噪声水平通常是索引，需要处理 'noise_x.xx' 格式
        try:
             noise_levels = [float(idx.split('_')[-1]) for idx in df_robustness.index]
             metric_values = df_robustness[metric_name].values
             # 排序以保证连线正确
             sorted_indices = np.argsort(noise_levels)
             noise_levels = np.array(noise_levels)[sorted_indices]
             metric_values = metric_values[sorted_indices]
             # (可选) 添加 noise=0 的点，需要原始指标
             # if initial_metrics and model_name in initial_metrics:
             #    noise_levels = np.insert(noise_levels, 0, 0.0)
             #    metric_values = np.insert(metric_values, 0, initial_metrics[model_name].get(metric_name, np.nan))
             plt.plot(noise_levels, metric_values, marker='o', linestyle='-', label=model_name)
        except Exception as e:
             print(f"Error processing or plotting robustness for {model_name}: {e}")
    plt.xlabel('Noise Level (Standard Deviation Ratio)', fontsize=12)
    plt.ylabel(f'{metric_name.upper()} (Lower is Better)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Robustness comparison plot for {metric_name} saved to {save_path}")

def plot_stability_comparison(results_df, metric_to_plot, title, save_path, plot_type='box'):
    """
    使用箱线图或小提琴图比较不同模型在多次运行中的性能指标分布。
    Args:
        results_df (pd.DataFrame): 包含多次运行结果的 DataFrame (由 main 函数生成)。
                                   列名应包含类似 'ModelName_metric' 的格式。
        metric_to_plot (str): 要绘制分布的指标名称 (e.g., 'mae', 'mse')。
        title (str): 图表标题。
        save_path (str): 图表保存路径。
        plot_type (str): 'box' 或 'violin'。
    """
    if results_df is None or results_df.empty:
        print("Warning: No results DataFrame provided for stability plotting.")
        return
    # 筛选出包含指定指标的列，并提取模型名称
    metric_cols = [col for col in results_df.columns if col.endswith(f'_{metric_to_plot}')]
    if not metric_cols:
        print(f"Warning: No columns found for metric '{metric_to_plot}' in results DataFrame.")
        return
    # 准备绘图所需的数据格式 (long format)
    plot_data = []
    for col in metric_cols:
        model_name = col.replace(f'_{metric_to_plot}', '')
        for value in results_df[col]:
            if pd.notna(value): # 确保值不是 NaN
                 plot_data.append({'Model': model_name, 'MetricValue': value})
    if not plot_data:
        print(f"Warning: No valid data found for metric '{metric_to_plot}' to plot stability.")
        return
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(max(10, len(metric_cols) * 1.5), 7)) # 动态调整宽度
    sns.set_palette("tab10")
    if plot_type == 'violin':
        sns.violinplot(x='Model', y='MetricValue', data=df_plot, inner='quartile', cut=0)
    else: # default to box plot
        sns.boxplot(x='Model', y='MetricValue', data=df_plot, showmeans=True) # showmeans 显示均值点
    plt.ylabel(f'{metric_to_plot.upper()} Distribution (Lower is Better)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Stability comparison plot for {metric_to_plot} saved to {save_path}")
def plot_full_dataset_predictions(data_dict, title, save_path, series_idx=0, max_points_per_set=500, target_cols_list=None):
    """
    绘制训练集、验证集、测试集的真实值与多个模型预测值的对比图。

    Args:
        data_dict (dict): 包含数据集信息的字典。结构如下:
            {
                'train': {'true': np.ndarray, 'predictions': {'ModelA': np.ndarray, 'ModelB': np.ndarray}},
                'val':   {'true': np.ndarray, 'predictions': {'ModelA': np.ndarray, 'ModelB': np.ndarray}},
                'test':  {'true': np.ndarray, 'predictions': {'ModelA': np.ndarray, 'ModelB': np.ndarray}}
            }
            其中 np.ndarray 的形状通常是 [num_samples, horizon, features]。
            如果某个数据集或模型的预测不存在，可以省略对应的键或值为 None。
        title (str): 图表标题。
        save_path (str): 图表保存路径。
        series_idx (int): 要绘制的目标序列的索引。
        max_points_per_set (int): 每个数据集（训练/验证/测试）最多绘制的点数。
    """
    plt.figure(figsize=(20, 10)) # 更大的画布
    colors = plt.cm.tab10(np.linspace(0, 1, 10)) # 预定义颜色
    model_styles = {name: ('--', colors[i % len(colors)], 0.7) for i, name in enumerate(
        data_dict.get('test', {}).get('predictions', {}).keys() # 基于测试集的模型列表确定样式
    )}
    set_colors = {'train': 'lightblue', 'val': 'lightgreen', 'test': 'lightcoral'}
    set_markers = {'train': '.', 'val': '.', 'test': '.'}
    set_true_styles = {'train': ('-', 'blue', 1.0), 'val': ('-', 'green', 1.0), 'test': ('-', 'red', 1.0)}

    current_offset = 0
    set_boundaries = {} # 记录每个数据集的结束位置

    if target_cols_list and series_idx < len(target_cols_list):
        target_col_name = target_cols_list[series_idx]
    else:
        target_col_name = f"Series {series_idx}"
    full_title = f"{title} - {target_col_name}"
    plt.title(full_title, fontsize=16)

    for set_name in ['train', 'val', 'test']:
        set_data = data_dict.get(set_name)
        if not set_data or set_data.get('true') is None:
            print(f"Info: No data found for '{set_name}' set. Skipping.")
            continue

        y_true = set_data['true']
        predictions = set_data.get('predictions', {})

        # --- 数据处理 ---
        if y_true.ndim != 3:
            print(f"Warning: True values for '{set_name}' have unexpected dimensions: {y_true.ndim}")
            continue
        try:
            y_true_flat = y_true[:, :, series_idx].reshape(-1)
        except IndexError:
            print(f"Error: series_idx {series_idx} is out of bounds for '{set_name}' true values features ({y_true.shape[-1]})")
            continue

        plot_len = min(max_points_per_set, len(y_true_flat))
        time_steps = np.arange(current_offset, current_offset + plot_len)

        # --- 绘制真实值 ---
        true_style = set_true_styles[set_name]
        plt.plot(time_steps, y_true_flat[:plot_len],
                 label=f'{set_name.capitalize()} True',
                 color=true_style[1], linestyle=true_style[0], linewidth=1.5, alpha=true_style[2],
                 marker=set_markers[set_name], markersize=3)

        # --- 绘制预测值 ---
        for model_name, y_pred in predictions.items():
            if y_pred is None: continue
            if y_pred.ndim != 3:
                print(f"Warning: Predictions for model '{model_name}' in '{set_name}' have unexpected dimensions: {y_pred.ndim}")
                continue
            try:
                y_pred_flat = y_pred[:, :, series_idx].reshape(-1)
                if len(y_pred_flat) != len(y_true_flat):
                     print(f"Warning: Length mismatch for '{model_name}' in '{set_name}' set.")
                     continue

                style = model_styles.get(model_name, ('--', 'gray', 0.6)) # 获取模型样式
                # 只为第一个数据集添加模型标签，避免图例重复
                label = f'{model_name} Preds' if set_name == 'train' else None
                plt.plot(time_steps, y_pred_flat[:plot_len],
                         label=label,
                         color=style[1], linestyle=style[0], alpha=style[2])
            except IndexError:
                print(f"Error: series_idx {series_idx} is out of bounds for '{model_name}' predictions in '{set_name}' ({y_pred.shape[-1]})")
            except Exception as e:
                 print(f"Error plotting predictions for {model_name} in {set_name}: {e}")

        current_offset += plot_len
        set_boundaries[set_name] = current_offset # 记录结束点

    # --- 添加数据集分割线和标签 ---
    last_boundary = 0
    for set_name, boundary in set_boundaries.items():
        if boundary > last_boundary: # 只有当该数据集被绘制时才添加
             plt.axvline(x=boundary, color='grey', linestyle=':', linewidth=1)
             # 在区域中间添加文本标签
             text_x = (last_boundary + boundary) / 2
             plt.text(text_x, plt.ylim()[1] * 0.95, set_name.upper(),
                      ha='center', va='top', backgroundcolor='white', alpha=0.8, fontsize=10)
             last_boundary = boundary


    plt.xlabel('Time Steps (Concatenated Train, Val, Test)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1)) # 图例放在外面
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例腾出空间
    plt.savefig(save_path)
    plt.close()
    print(f"Full dataset prediction plot saved to {save_path}")

def save_plot(plt_obj, filepath, logger):
    """
    保存 matplotlib 图表到文件。
    Args:
        plt_obj: matplotlib.pyplot 对象。
        filepath (str): 图表保存路径。
        logger (logging.Logger): 日志记录器实例。
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt_obj.savefig(filepath)
        plt_obj.close()
        logger.info(f"Plot saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving plot to {filepath}: {e}")

def plot_weights_biases_distribution(model, save_dir, model_name="Model"):
    """
    绘制模型权重和偏置的分布直方图。
    Args:
        model (torch.nn.Module): PyTorch 模型。
        save_dir (str): 图表保存的目录。
        model_name (str): 模型名称，用于图表标题。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'{model_name} Weights and Biases Distribution', fontsize=16)
    
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            param_count += 1
            # 限制只绘制前20个参数，避免图表过大
            if param_count > 20:
                logging.warning(f"Plotting only first 20 parameters for {model_name}. Too many parameters to plot.")
                break
            
            plt.subplot(4, 5, param_count) # 假设最多20个子图
            param_data = param.data.cpu().numpy().flatten()
            
            sns.histplot(param_data, kde=True, ax=plt.gca())
            plt.title(name, fontsize=8)
            plt.xlabel('') # 移除 x 轴标签，避免拥挤
            plt.ylabel('') # 移除 y 轴标签，避免拥挤
            plt.tick_params(axis='both', which='major', labelsize=6) # 调整刻度标签大小
            
    if param_count == 0:
        logging.info(f"No trainable parameters found in {model_name} to plot distribution.")
        plt.close() # 关闭空图
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应suptitle
    plot_path = os.path.join(save_dir, f'{model_name.lower()}_weights_biases_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Weights and biases distribution plot saved to {plot_path}")

def save_results_to_csv(df, filepath, logger):
    """
    将 DataFrame 保存到 CSV 文件。
    Args:
        df (pd.DataFrame): 要保存的 DataFrame。
        filepath (str): CSV 文件的完整路径。
        logger (logging.Logger): 日志记录器实例。
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")

def plot_acf_pacf(residuals, save_dir, model_name="Model", series_idx=0, target_cols_list=None, lags=40):
    """
    绘制残差的自相关函数 (ACF) 和偏自相关函数 (PACF) 图。
    Args:
        residuals (np.ndarray): 残差数组 (一维)。
        save_dir (str): 图表保存的目录。
        model_name (str): 模型名称，用于图表标题。
        series_idx (int): 对应的目标序列索引。
        target_cols_list (list): 目标列名称列表。
        lags (int): 要计算和绘制的最大滞后数。
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        logging.error("statsmodels library not found. Please install it: pip install statsmodels")
        return

    os.makedirs(save_dir, exist_ok=True)

    if target_cols_list and series_idx < len(target_cols_list):
        target_col_name = target_cols_list[series_idx]
    else:
        target_col_name = f"Series {series_idx}"

    # ACF 图
    fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
    plot_acf(residuals, lags=lags, ax=ax_acf, title=f'{model_name} Residuals ACF - {target_col_name}')
    acf_path = os.path.join(save_dir, f'{model_name.lower()}_{target_col_name.lower().replace(" ", "_")}_residuals_acf.png')
    fig_acf.tight_layout()
    fig_acf.savefig(acf_path)
    plt.close(fig_acf)
    logging.info(f"Residuals ACF plot saved to {acf_path}")

    # PACF 图
    fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
    plot_pacf(residuals, lags=lags, ax=ax_pacf, title=f'{model_name} Residuals PACF - {target_col_name}')
    pacf_path = os.path.join(save_dir, f'{model_name.lower()}_{target_col_name.lower().replace(" ", "_")}_residuals_pacf.png')
    fig_pacf.tight_layout()
    fig_pacf.savefig(pacf_path)
    plt.close(fig_pacf)
    logging.info(f"Residuals PACF plot saved to {pacf_path}")

def save_experiment_metadata(config_obj, save_dir, experiment_id):
    """
    保存实验的元数据，包括配置参数。
    Args:
        config_obj (Config): 配置对象实例。
        save_dir (str): 元数据保存的目录。
        experiment_id (str): 实验的唯一标识符。
    """
    os.makedirs(save_dir, exist_ok=True)
    metadata_path = os.path.join(save_dir, f'{experiment_id}_metadata.json')
    
    metadata = {}
    # 遍历 Config 对象的所有属性
    for attr_name in dir(config_obj):
        if not attr_name.startswith('__') and not callable(getattr(config_obj, attr_name)):
            attr_value = getattr(config_obj, attr_name)
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
    
    # 移除一些不必要的或敏感的路径信息，或者在保存前进行处理
    if 'BASE_DIR' in metadata:
        del metadata['BASE_DIR']
    if 'LOG_FILE_PATH' in metadata:
        del metadata['LOG_FILE_PATH']
    
    try:
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Experiment metadata saved to {metadata_path}")
    except Exception as e:
        logging.error(f"Error saving experiment metadata to {metadata_path}: {e}")

def plot_error_distribution(y_true, y_pred, save_dir, model_name="Model", series_idx=0, target_cols_list=None, plot_type='box'):
    """
    绘制预测误差的箱线图或小提琴图。
    Args:
        y_true (np.ndarray): 真实值数组 [num_samples, horizon, features]。
        y_pred (np.ndarray): 预测值数组 [num_samples, horizon, features]。
        save_dir (str): 图表保存的目录。
        model_name (str): 模型名称，用于图表标题。
        series_idx (int): 要分析的目标序列的索引。
        target_cols_list (list): 目标列名称列表。
        plot_type (str): 'box' 或 'violin'。
    """
    os.makedirs(save_dir, exist_ok=True)

    if y_true.ndim != 3 or y_pred.ndim != 3:
        logging.warning(f"Unexpected data dimensions for error distribution. True: {y_true.ndim}, Pred: {y_pred.ndim}. Expected 3D [samples, horizon, features]. Skipping.")
        return
    if y_true.shape != y_pred.shape:
        logging.warning(f"Shape mismatch between true values {y_true.shape} and predictions {y_pred.shape}. Cannot perform error distribution analysis. Skipping.")
        return
    if y_true.shape[2] <= series_idx:
        logging.warning(f"Series index {series_idx} is out of bounds for available features ({y_true.shape[2]}). Skipping error distribution analysis.")
        return

    # 提取指定序列的数据并展平
    y_true_flat = y_true[:, :, series_idx].reshape(-1)
    y_pred_flat = y_pred[:, :, series_idx].reshape(-1)
    errors = np.abs(y_true_flat - y_pred_flat) # 绝对误差

    if target_cols_list and series_idx < len(target_cols_list):
        target_col_name = target_cols_list[series_idx]
    else:
        target_col_name = f"Series {series_idx}"

    plt.figure(figsize=(10, 6))
    sns.set_palette("pastel")

    if plot_type == 'violin':
        sns.violinplot(y=errors, inner='quartile', cut=0)
        plt.title(f'{model_name} Absolute Error Distribution (Violin Plot) - {target_col_name}')
    else: # default to box plot
        sns.boxplot(y=errors, showmeans=True)
        plt.title(f'{model_name} Absolute Error Distribution (Box Plot) - {target_col_name}')
    
    plt.ylabel('Absolute Error Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    error_plot_path = os.path.join(save_dir, f'{model_name.lower()}_{target_col_name.lower().replace(" ", "_")}_error_distribution_{plot_type}.png')
    plt.savefig(error_plot_path)
    plt.close()
    logging.info(f"Error distribution ({plot_type}) plot saved to {error_plot_path}")