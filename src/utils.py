import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from src import config
import seaborn as sns 

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

def plot_predictions(y_true, y_pred, title, save_path, num_series_to_plot=1, series_idx=0):
    """绘制真实值与预测值的对比图 (只绘制部分序列和样本)"""
    if y_true.ndim == 3: # [num_samples, horizon, features] -> [num_samples*horizon, features]
        y_true_flat = y_true[:, :, series_idx].reshape(-1)
        y_pred_flat = y_pred[:, :, series_idx].reshape(-1)
    elif y_true.ndim == 2: # [num_samples, horizon] - assume single feature
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
    else:
        print(f"Warning: Unexpected data dimensions for plotting: {y_true.ndim}")
        return

    plt.figure(figsize=(15, 7))
    # 只绘制前 500 个点以保持清晰
    plot_len = min(500, len(y_true_flat))
    plt.plot(y_true_flat[:plot_len], label='True Values', marker='.', linestyle='-')
    plt.plot(y_pred_flat[:plot_len], label='Predictions', marker='x', linestyle='--')

    # 如果提供了 TARGET_COLS，使用具体的列名
    target_col_name = config.TARGET_COLS[series_idx] if series_idx < len(config.TARGET_COLS) else f"Series {series_idx}"
    full_title = f"{title} - {target_col_name}"

    plt.title(full_title)
    plt.xlabel('Time Steps (Sample Index)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot saved to {save_path}")

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

def plot_comparison_predictions(true_values, predictions_dict, title, save_path, series_idx=0, max_points=500):
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
    target_col_name = config.TARGET_COLS[series_idx] if series_idx < len(config.TARGET_COLS) else f"Series {series_idx}"
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