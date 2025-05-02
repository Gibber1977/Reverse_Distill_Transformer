import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src import config, utils
from tqdm import tqdm
import pandas as pd
import os

def calculate_metrics(y_true, y_pred):
    """计算 MSE 和 MAE"""
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
    if 'mse' in config.METRICS:
        metrics['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
    if 'mae' in config.METRICS:
        metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)

    # (可选) 可以添加每个特征的指标
    # for i in range(y_true_flat.shape[1]):
    #     col_name = config.TARGET_COLS[i] if i < len(config.TARGET_COLS) else f"feature_{i}"
    #     metrics[f'mse_{col_name}'] = mean_squared_error(y_true_flat[:, i], y_pred_flat[:, i])
    #     metrics[f'mae_{col_name}'] = mean_absolute_error(y_true_flat[:, i], y_pred_flat[:, i])

    return metrics

def predict(model, dataloader, device):
    """使用模型进行预测"""
    model.eval()
    model.to(device)
    all_preds = []
    all_trues = []
    with torch.no_grad():
        predict_iterator = tqdm(dataloader, desc="Predicting", leave=False)
        for batch_x, batch_y in predict_iterator:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            input_dict = {'insample_y': batch_x}
            outputs = model(input_dict)
            
            all_preds.append(outputs.cpu())
            all_trues.append(batch_y.cpu())

    # 将列表中的批次连接起来
    predictions = torch.cat(all_preds, dim=0)
    true_values = torch.cat(all_trues, dim=0)
    return true_values, predictions # 返回 torch tensors

def evaluate_model(model, dataloader, device, scaler, model_name="Model", plots_dir="."):
    """在测试集上评估模型性能"""
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
    metrics = calculate_metrics(true_values_original, predictions_original)
    print(f"Evaluation Metrics for {model_name} (Original Scale):")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.6f}")

    # --- 保存预测结果和真实值 (可选，用于详细分析) ---
    # np.save(os.path.join(config.RESULTS_DIR, f"{model_name}_preds.npy"), predictions_original)
    # np.save(os.path.join(config.RESULTS_DIR, f"{model_name}_trues.npy"), true_values_original)

    # --- 绘制预测图 (只绘制第一个特征) ---
    plot_save_path = os.path.join(plots_dir, f"{model_name}_test_predictions.png")
    utils.plot_predictions(true_values_original, predictions_original,
                           title=f"{model_name} - Test Set Predictions vs True Values",
                           save_path=plot_save_path, series_idx=0)

    return metrics, true_values_original, predictions_original


def evaluate_robustness(model, dataloader, device, scaler, noise_levels, model_name="Model", metrics_dir="."):
    """评估模型在不同噪声水平下的鲁棒性"""
    print(f"\n--- Evaluating Robustness for {model_name} ---")
    robustness_results = {}

    # 获取原始的、干净的测试集预测和真实值 (用于比较)
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
            for batch_x, batch_y in noisy_iterator:
                # --- STEP 1: Move input data to the target device FIRST ---
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) # Also move labels if needed later on device

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
        metrics_noisy = calculate_metrics(true_values_original_noisy, predictions_original_noisy)
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
