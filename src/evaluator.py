import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src import config, utils
from tqdm import tqdm
import pandas as pd
import os
from scipy.spatial.distance import cosine

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度。
    vec1, vec2: numpy array 或 torch.Tensor
    """
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().numpy()

    # 展平向量
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()

    # 检查是否为零向量，避免除以零
    if np.all(vec1_flat == 0) or np.all(vec2_flat == 0):
        return 0.0 # 或根据业务逻辑返回其他值，例如 NaN

    return 1 - cosine(vec1_flat, vec2_flat) # scipy.spatial.distance.cosine 返回余弦距离，1 - 距离 = 相似度

def calculate_similarity_metrics(student_preds, teacher_preds, metric_type):
    """
    计算学生模型和教师模型预测结果之间的相似度指标。
    student_preds: 学生模型的预测结果 (torch.Tensor 或 numpy array)
    teacher_preds: 教师模型的预测结果 (torch.Tensor 或 numpy array)
    metric_type: 字符串, 相似度指标类型 ('cosine_similarity', 'euclidean_distance')
    """
    if isinstance(student_preds, torch.Tensor):
        student_preds = student_preds.cpu().numpy()
    if isinstance(teacher_preds, torch.Tensor):
        teacher_preds = teacher_preds.cpu().numpy()

    similarity_score = 0.0
    if metric_type == 'cosine_similarity':
        similarity_score = cosine_similarity(student_preds, teacher_preds)
    elif metric_type == 'euclidean_distance':
        # 欧几里得距离，越小越相似
        similarity_score = np.linalg.norm(student_preds - teacher_preds)
    else:
        raise ValueError(f"Unsupported similarity metric type: {metric_type}")

    return {f'similarity_{metric_type}': similarity_score}

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """计算 MAPE，处理分母为零的情况"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    non_zero_mask = np.abs(y_true) > epsilon
    if np.sum(non_zero_mask) == 0:
        return np.nan # 如果所有真实值都接近零，无法计算 MAPE
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def weighted_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """计算 WAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    absolute_error = np.abs(y_true - y_pred)
    sum_absolute_error = np.sum(absolute_error)
    sum_absolute_true = np.sum(np.abs(y_true))
    if sum_absolute_true < epsilon:
        return np.nan # 如果所有真实值的绝对值之和接近零
    return (sum_absolute_error / sum_absolute_true) * 100


def calculate_metrics(y_true, y_pred, metrics_list):
    """计算 MSE, MAE, MAPE, WAPE"""
    # 确保是 numpy 数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # 展平数据以便计算指标 [samples*horizon, features] or [samples*horizon]
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    metrics = {}
    # 逐个特征计算或整体计算
    # 这里我们计算整体指标（先展平再计算）
    if 'mse' in metrics_list:
        metrics['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
    if 'mae' in metrics_list:
        metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)

    if 'mape' in metrics_list: # 假设 metrics_list 会包含 'mape'
        metrics['mape'] = mean_absolute_percentage_error(y_true_flat, y_pred_flat)
    if 'wape' in metrics_list: # 假设 metrics_list 会包含 'wape'
        metrics['wape'] = weighted_absolute_percentage_error(y_true_flat, y_pred_flat)

    # (可选) 可以添加每个特征的指标
    # for i in range(y_true_flat.shape[1]):
    #     col_name = config.TARGET_COLS[i] if i < len(config.TARGET_COLS) else f"feature_{i}"
    #     metrics[f'mse_{col_name}'] = mean_squared_error(y_true_flat[:, i], y_pred_flat[:, i])
    #     metrics[f'mae_{col_name}'] = mean_absolute_error(y_true_flat[:, i], y_pred_flat[:, i])

    return metrics

def predict(model, dataloader, device):
    """使用模型进行预测"""
    # 确保 model 是一个 PyTorch 模型实例
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected 'model' to be a PyTorch model (torch.nn.Module), but received type: {type(model)}. Please ensure a valid PyTorch model is passed.")

    # 将模型设置为评估模式并移动到指定设备
    model.eval()
    model.to(device)

    # 初始化列表以存储所有预测和真实值
    all_preds = []
    all_trues = []

    # 在不计算梯度的模式下进行预测
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        predict_iterator = tqdm(dataloader, desc="Predicting", leave=False)
        for batch_x, batch_y, batch_hist_exog, batch_futr_exog in predict_iterator:
            # 将输入数据移动到指定设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 构建模型输入字典
            input_dict = {'insample_y': batch_x}
            if batch_hist_exog is not None:
                batch_hist_exog = batch_hist_exog.to(device)
                input_dict['hist_exog'] = batch_hist_exog
            else:
                input_dict['hist_exog'] = None
            if batch_futr_exog is not None:
                batch_futr_exog = batch_futr_exog.to(device)
                input_dict['futr_exog'] = batch_futr_exog
            else:
                input_dict['futr_exog'] = None

            # 执行模型前向传播
            outputs = model(input_dict)

            # 将预测结果和真实值从设备移动到 CPU 并添加到列表中
            all_preds.append(outputs.cpu())
            all_trues.append(batch_y.cpu())

    # 将所有批次的预测结果和真实值连接起来
    predictions = torch.cat(all_preds, dim=0)
    true_values = torch.cat(all_trues, dim=0)

    # 返回真实值和预测结果
    return true_values, predictions # 返回 torch tensors

def evaluate_model(model, dataloader, device, scaler, config_obj, model_name="Model", plots_dir=".", teacher_predictions_original=None):
    """
    在测试集上评估模型性能，并可选地计算学生-教师模型相似度。
    model: 要评估的模型 (学生模型)
    dataloader: 测试数据加载器
    device: 计算设备
    scaler: 用于逆变换的 StandardScaler
    model_name: 模型名称
    plots_dir: 绘图保存目录
    teacher_predictions_original: 可选，教师模型在相同数据上的原始尺度预测结果 (numpy array)，用于计算相似度
    """
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    true_values_scaled, predictions_scaled = predict(model, dataloader, device)

    # --- 逆变换回原始尺度 ---
    # scaler.inverse_transform 期望 [n_samples, n_features]
    # 需要重塑数据：[batch, horizon, features] -> [batch*horizon, features]
    n_samples, horizon, n_features = predictions_scaled.shape
    pred_reshaped = predictions_scaled.view(-1, n_features).numpy()
    true_reshaped = true_values_scaled.view(-1, n_features).numpy()

    try:
        predictions_original = scaler.inverse_transform(pred_reshaped)
        true_values_original = scaler.inverse_transform(true_reshaped)
    except Exception as e:
        print(f"Error during inverse transform: {e}")
        print("Using scaled values for metric calculation.")
        predictions_original = pred_reshaped
        true_values_original = true_reshaped


    # 重塑回 [batch, horizon, features] 供绘图和原始指标计算
    predictions_original = predictions_original.reshape(n_samples, horizon, n_features)
    true_values_original = true_values_original.reshape(n_samples, horizon, n_features)

    # --- 计算指标 ---
    # --- 计算指标 ---
    metrics = calculate_metrics(true_values_original, predictions_original, config_obj.METRICS)
    print(f"Evaluation Metrics for {model_name} (Original Scale):")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.6f}")

    # --- 计算学生-教师模型相似度 (如果提供了教师预测) ---
    if teacher_predictions_original is not None:
        print(f"\n--- Calculating Student-Teacher Similarity ({config_obj.SIMILARITY_METRIC}) ---")
        similarity_metrics = calculate_similarity_metrics(predictions_original, teacher_predictions_original, config_obj.SIMILARITY_METRIC)
        metrics.update(similarity_metrics)
        for key, value in similarity_metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.6f}")

    # --- 保存预测结果和真实值 (可选，用于详细分析) ---
    # np.save(os.path.join(config.RESULTS_DIR, f"{model_name}_preds.npy"), predictions_original)
    # np.save(os.path.join(config.RESULTS_DIR, f"{model_name}_trues.npy"), true_values_original)

    # --- 绘制预测图 (只绘制第一个特征) ---
    plot_save_path = os.path.join(plots_dir, f"{model_name}_test_predictions.png")
    utils.plot_predictions(true_values_original, predictions_original,
                           title=f"{model_name} - Test Set Predictions vs True Values",
                           save_path=plot_save_path, series_idx=0,
                           target_cols_list=config_obj.TARGET_COLS)

    return metrics, true_values_original, predictions_original

    # --- 绘制残差分析图 ---
    # 提取第一个特征的残差进行 ACF/PACF 分析
    residuals_flat = (true_values_original[:, :, 0] - predictions_original[:, :, 0]).flatten()
    utils.plot_residuals_analysis(true_values_original, predictions_original,
                                  save_dir=plots_dir, model_name=model_name,
                                  series_idx=0, target_cols_list=config_obj.TARGET_COLS)
    utils.plot_acf_pacf(residuals_flat, save_dir=plots_dir, model_name=model_name,
                        series_idx=0, target_cols_list=config_obj.TARGET_COLS)
    utils.plot_error_distribution(true_values_original, predictions_original,
                                  save_dir=plots_dir, model_name=model_name,
                                  series_idx=0, target_cols_list=config_obj.TARGET_COLS,
                                  plot_type='box') # 默认绘制箱线图
    utils.plot_error_distribution(true_values_original, predictions_original,
                                  save_dir=plots_dir, model_name=model_name,
                                  series_idx=0, target_cols_list=config_obj.TARGET_COLS,
                                  plot_type='violin') # 绘制小提琴图


def evaluate_robustness(model, dataloader, device, scaler, noise_levels, config_obj, model_name="Model", metrics_dir="."):
    """评估模型在不同噪声水平下的鲁棒性"""
    print(f"\n--- Evaluating Robustness for {model_name} ---")
    robustness_results = {}

    # 获取原始的、干净的测试集预测和真实值 (用于比较)
    # predict 函数现在返回 (true_values, predictions)
    original_trues_scaled, original_preds_scaled = predict(model, dataloader, device)

    # ... (inside evaluate_robustness function) ...

    for noise_level in noise_levels:
        print(f"  Testing with noise level (std ratio): {noise_level}")
        noisy_preds_list = []
        noisy_trues_list = [] # 真实值也需要保留对应关系

        model.eval()
        model.to(device) # 确保模型在设备上
        with torch.no_grad():
            noisy_iterator = tqdm(dataloader, desc=f"Predicting (Noise={noise_level})", leave=False)
            for batch_x, batch_y, batch_hist_exog, batch_futr_exog in noisy_iterator:
                # --- STEP 1: Move input data to the target device FIRST ---
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) # Also move labels if needed later on device
                # Move exog variables if they exist
                if batch_hist_exog is not None:
                    batch_hist_exog = batch_hist_exog.to(device)
                if batch_futr_exog is not None:
                    batch_futr_exog = batch_futr_exog.to(device)

                # --- STEP 2: Calculate noise based on data ON THE DEVICE ---
                # Calculate std dev on the device itself
                noise_std = batch_x.std(dim=(0, 1), keepdim=True) * noise_level
                # randn_like will create noise on the SAME device as batch_x (which is now `device`)
                noise = torch.randn_like(batch_x) * noise_std

                # --- STEP 3: Perform the addition - BOTH tensors should now be on `device` ---
                # NO .to(device) needed for noise here! It should already be on the device.
                noisy_batch_x = batch_x + noise

                # --- STEP 4: Model prediction (Input must be on the model's device) ---
                input_dict = {'insample_y': noisy_batch_x}
                # 始终添加 exog 键，如果加载器未提供则设为 None
                # 注意：噪声是加在 batch_x 上的，外生变量通常不加噪声
                input_dict['hist_exog'] = batch_hist_exog if batch_hist_exog is not None else None # Already on device or None
                input_dict['futr_exog'] = batch_futr_exog if batch_futr_exog is not None else None # Already on device or None

                outputs = model(input_dict)

                # --- STEP 5: Move results back to CPU for storage/aggregation ---
                noisy_preds_list.append(outputs.cpu())
                noisy_trues_list.append(batch_y.cpu()) # Move labels back too

        # --- Concatenation (on CPU) ---
        noisy_predictions_scaled = torch.cat(noisy_preds_list, dim=0)
        noisy_true_values_scaled = torch.cat(noisy_trues_list, dim=0)

        # --- Inverse Transform (on CPU using numpy) ---
        n_samples, horizon, n_features = noisy_predictions_scaled.shape
        pred_reshaped_noisy = noisy_predictions_scaled.view(-1, n_features).numpy()
        true_reshaped_noisy = noisy_true_values_scaled.view(-1, n_features).numpy()

        try:
            predictions_original_noisy = scaler.inverse_transform(pred_reshaped_noisy)
            true_values_original_noisy = scaler.inverse_transform(true_reshaped_noisy)
        except Exception as e:
            print(f"Error during inverse transform (noisy data): {e}. Using scaled.")
            predictions_original_noisy = pred_reshaped_noisy
            true_values_original_noisy = true_reshaped_noisy

        # Reshape back if needed for consistency, still numpy arrays on CPU
        predictions_original_noisy = predictions_original_noisy.reshape(n_samples, horizon, n_features)
        true_values_original_noisy = true_values_original_noisy.reshape(n_samples, horizon, n_features)


        # --- Calculate Metrics (on CPU using numpy) ---
        metrics_noisy = calculate_metrics(true_values_original_noisy, predictions_original_noisy, config_obj.METRICS)
        robustness_results[f"noise_{noise_level}"] = metrics_noisy
        print(f"  Metrics at noise={noise_level}: {metrics_noisy}")

    # --- DataFrame creation and saving (on CPU) ---
    df_robustness = pd.DataFrame(robustness_results).T # 转置使噪声水平为行
    df_robustness.index.name = 'Noise Level (std ratio)'
    save_path = os.path.join(metrics_dir, f"{model_name}_robustness.csv")
    df_robustness.to_csv(save_path)
    print(f"Robustness results saved to {save_path}")

    return df_robustness


# 稳定性评估需要在 main 脚本中通过多次运行实现
