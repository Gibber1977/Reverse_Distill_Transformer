import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity # 重命名以避免与现有函数冲突
from src import config, utils
from tqdm import tqdm
import pandas as pd
import os
from scipy.spatial.distance import cosine # 现有自定义cosine_similarity使用它

# 现有的自定义 cosine_similarity 函数保持不变
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


def calculate_error_cosine_similarity(y_true, pred1, pred2):
  """
  计算两个模型预测误差的余弦相似度。
  误差定义为: error = y_true - pred。
  """
  y_true = np.asarray(y_true)
  pred1 = np.asarray(pred1)
  pred2 = np.asarray(pred2)

  error1 = y_true - pred1
  error2 = y_true - pred2

  # 确保误差是二维的，即使只有一个特征
  error1_flat = error1.reshape(error1.shape[0]*error1.shape[1], -1) if error1.ndim > 2 else error1.reshape(error1.shape[0], -1)
  error2_flat = error2.reshape(error2.shape[0]*error2.shape[1], -1) if error2.ndim > 2 else error2.reshape(error2.shape[0], -1)
  
  # 将所有样本的误差展平为一个长向量进行比较
  error1_flat_overall = error1_flat.flatten().reshape(1, -1)
  error2_flat_overall = error2_flat.flatten().reshape(1, -1)

  if error1_flat_overall.shape[1] == 0 or error2_flat_overall.shape[1] == 0:
    return np.nan # 如果没有数据点

  # 如果两个误差向量都为零，则它们是完全相似的
  if np.all(error1_flat_overall == 0) and np.all(error2_flat_overall == 0):
    return 1.0
  # 如果只有一个误差向量为零，则它们不相似（除非另一个也为零，已处理）
  elif np.all(error1_flat_overall == 0) or np.all(error2_flat_overall == 0):
    return 0.0
  
  similarity = sklearn_cosine_similarity(error1_flat_overall, error2_flat_overall)
  return similarity[0, 0]


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

def predict(model, dataloader, device, config_obj): # 添加 config_obj 参数
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
            batch_x_device, batch_y_device = batch_x.to(device), batch_y.to(device) # 使用新变量名以避免混淆原始 batch_x

            # Split input_x for DLinear compatibility
            insample_y = batch_x_device[:, :, :len(config_obj.TARGET_COLS)]
            X_df_batch = batch_x_device[:, :, len(config_obj.TARGET_COLS):]
            if X_df_batch.shape[2] == 0: # If no exogenous features
                X_df_batch = None
            
            input_dict = {'insample_y': insample_y, 'X_df': X_df_batch}
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
            target_y_to_append = batch_y_device[:, :, :len(config_obj.TARGET_COLS)]
            all_trues.append(target_y_to_append.cpu())

    # 将所有批次的预测结果和真实值连接起来
    predictions = torch.cat(all_preds, dim=0)
    true_values = torch.cat(all_trues, dim=0)

    # 返回真实值和预测结果
    return true_values, predictions # 返回 torch tensors

def evaluate_model(model, dataloader, device, scaler, config_obj, logger, model_name="Model", teacher_predictions_original=None, dataset_type="Test Set"):
    """
    在指定数据集上评估模型性能，并可选地计算学生-教师模型相似度。
    model: 要评估的模型 (学生模型)
    dataloader: 数据加载器 (可以是验证集或测试集)
    device: 计算设备
    scaler: 用于逆变换的 StandardScaler
    config_obj: 配置对象
    logger: 日志记录器
    model_name: 模型名称
    teacher_predictions_original: 可选，教师模型在相同数据上的原始尺度预测结果 (numpy array)，用于计算相似度
    dataset_type: 字符串，表示正在评估的数据集类型 (例如 "Validation Set" 或 "Test Set")
    """
    logger.info(f"\n--- Evaluating {model_name} on {dataset_type} ---")
    true_values_scaled, predictions_scaled = predict(model, dataloader, device, config_obj) # 传递 config_obj

    # --- 逆变换回原始尺度 ---
    n_samples, horizon, n_scaled_features = predictions_scaled.shape # n_scaled_features is len(TARGET_COLS)
    
    # Helper function for inverse transform
    def _inverse_transform_target_cols(scaled_data, scaler_obj, config):
        # scaled_data shape: (num_samples * horizon, len(TARGET_COLS))
        # scaler_obj was fit on N_FEATURES
        dummy_data_for_inverse = np.zeros((scaled_data.shape[0], config.N_FEATURES))
        dummy_data_for_inverse[:, :len(config.TARGET_COLS)] = scaled_data
        original_all_features = scaler_obj.inverse_transform(dummy_data_for_inverse)
        original_target_cols = original_all_features[:, :len(config.TARGET_COLS)]
        return original_target_cols

    try:
        # Reshape predictions_scaled to (num_samples * horizon, len(TARGET_COLS))
        predictions_reshaped_scaled = predictions_scaled.view(-1, n_scaled_features).cpu().numpy()
        predictions_original = _inverse_transform_target_cols(predictions_reshaped_scaled, scaler, config_obj)

        # Reshape true_values_scaled to (num_samples * horizon, len(TARGET_COLS))
        true_values_reshaped_scaled = true_values_scaled.view(-1, n_scaled_features).cpu().numpy()
        true_values_original = _inverse_transform_target_cols(true_values_reshaped_scaled, scaler, config_obj)

    except Exception as e:
        logger.error(f"Error during inverse transform: {e}")
        logger.warning("Using scaled values for metric calculation.")
        # Fallback to scaled values if inverse transform fails
        predictions_original = predictions_scaled.view(-1, n_scaled_features).cpu().numpy()
        true_values_original = true_values_scaled.view(-1, n_scaled_features).cpu().numpy()

    # 重塑回 [batch, horizon, len(TARGET_COLS)]
    n_features_to_reshape = len(config_obj.TARGET_COLS)
    predictions_original = predictions_original.reshape(n_samples, horizon, n_features_to_reshape)
    true_values_original = true_values_original.reshape(n_samples, horizon, n_features_to_reshape)

    # At this point, true_values_original and predictions_original already contain only TARGET_COLS
    # So, the explicit slicing below is redundant but harmless.
    # true_values_original = true_values_original[:, :, :len(config_obj.TARGET_COLS)]
    # predictions_original = predictions_original[:, :, :len(config_obj.TARGET_COLS)]

    # --- 计算指标 ---
    metrics = calculate_metrics(true_values_original, predictions_original, config_obj.METRICS)
    logger.info(f"Evaluation Metrics for {model_name} (Original Scale):")
    for key, value in metrics.items():
        logger.info(f"  {key.upper()}: {value:.6f}")

    # --- 计算学生-教师模型相似度 (如果提供了教师预测) ---
    similarity_metrics = {} # 初始化 similarity_metrics 字典
    if teacher_predictions_original is not None:
        logger.info(f"\n--- Calculating Student-Teacher Similarity ({config_obj.SIMILARITY_METRIC}) ---")
        # 原有的相似度计算
        try:
            standard_similarity_metrics = calculate_similarity_metrics(predictions_original, teacher_predictions_original, config_obj.SIMILARITY_METRIC)
            similarity_metrics.update(standard_similarity_metrics)
            for key, value in standard_similarity_metrics.items():
                logger.info(f"  {key.replace('_', ' ').title()}: {value:.6f}")
        except Exception as e:
            logger.error(f"Error calculating standard similarity: {e}")
            # 确保即使出错，键也存在
            similarity_metrics[f'similarity_{config_obj.SIMILARITY_METRIC}'] = np.nan


        # 计算 error_cos_similarity
        # true_values_original, predictions_original (model1_pred), teacher_predictions_original (model2_pred)
        if true_values_original is not None and predictions_original is not None: # teacher_predictions_original 已经在外部 if 中检查
            try:
                error_cos_sim = calculate_error_cosine_similarity(true_values_original, predictions_original, teacher_predictions_original)
                similarity_metrics['error_cos_similarity'] = error_cos_sim
                logger.info(f"  Error Cosine Similarity: {error_cos_sim:.6f}")
            except Exception as e:
                logger.error(f"Error calculating error_cos_similarity: {e}")
                similarity_metrics['error_cos_similarity'] = np.nan
        else:
            similarity_metrics['error_cos_similarity'] = np.nan
    else:
        # 如果 teacher_predictions_original 为 None，所有基于它的相似度指标都应为 NaN
        similarity_metrics[f'similarity_{config_obj.SIMILARITY_METRIC}'] = np.nan # 确保原有指标键存在
        similarity_metrics['error_cos_similarity'] = np.nan

    metrics.update(similarity_metrics) # 将所有计算得到的相似度指标（包括新的和旧的）合并到主 metrics 字典中

    # --- 保存预测结果和真实值 (可选，用于详细分析) ---
    # np.save(os.path.join(config.RESULTS_DIR, f"{model_name}_preds.npy"), predictions_original)
    # np.save(os.path.join(config.RESULTS_DIR, f"{model_name}_trues.npy"), true_values_original)

    # --- 绘制预测图 (只绘制第一个特征) ---
    # 构建更详细的图片文件名和标题
    # 从 config_obj 中获取实验参数
    dataset_name = config_obj.DATASET_NAME if hasattr(config_obj, 'DATASET_NAME') else "UnknownDataset"
    pred_horizon = config_obj.PREDICTION_HORIZON
    noise_level = config_obj.TRAIN_NOISE_INJECTION_LEVEL
    smoothing_weight_smoothing = config_obj.SMOOTHING_WEIGHT_SMOOTHING
    run_idx = config_obj.RUN_IDX if hasattr(config_obj, 'RUN_IDX') else "UnknownRun"
    current_seed = config_obj.SEED if hasattr(config_obj, 'SEED') else "UnknownSeed"

    # 确保 plots_dir 是 config.RESULTS_DIR 下的 plots 子目录
    actual_plots_dir = os.path.join(config_obj.RESULTS_DIR, "plots")
    os.makedirs(actual_plots_dir, exist_ok=True) # 确保目录存在

    plot_filename = (
        f"{model_name}_"
        f"{dataset_name}_h{pred_horizon}_noise{noise_level}_smooth_w{smoothing_weight_smoothing}_"
        f"run{run_idx}_seed{current_seed}_test_predictions.png"
    )
    plot_save_path = os.path.join(actual_plots_dir, plot_filename)

    plot_title = (
        f"{model_name} - {dataset_name} (H:{pred_horizon}, Noise:{noise_level}, Smooth:{smoothing_weight_smoothing}) "
        f"Run {run_idx} Seed {current_seed} - Test Set Predictions vs True Values"
    )

    utils.plot_predictions(true_values_original, predictions_original,
                           title=plot_title,
                           save_path=plot_save_path, series_idx=0,
                           target_cols_list=config_obj.TARGET_COLS)

    # --- 绘制残差分析图 ---
    # 提取第一个特征的残差进行 ACF/PACF 分析
    residuals_flat = (true_values_original[:, :, 0] - predictions_original[:, :, 0]).flatten()
    utils.plot_residuals_analysis(true_values_original, predictions_original,
                                  save_dir=actual_plots_dir, model_name=model_name,
                                  series_idx=0, target_cols_list=config_obj.TARGET_COLS)
    utils.plot_acf_pacf(residuals_flat, save_dir=actual_plots_dir, model_name=model_name,
                        series_idx=0, target_cols_list=config_obj.TARGET_COLS)
    utils.plot_error_distribution(true_values_original, predictions_original,
                                  save_dir=actual_plots_dir, model_name=model_name,
                                  series_idx=0, target_cols_list=config_obj.TARGET_COLS,
                                  plot_type='box') # 默认绘制箱线图
    utils.plot_error_distribution(true_values_original, predictions_original,
                                  save_dir=actual_plots_dir, model_name=model_name,
                                  series_idx=0, target_cols_list=config_obj.TARGET_COLS,
                                  plot_type='violin') # 绘制小提琴图

    return metrics, true_values_original, predictions_original


def evaluate_robustness(model, dataloader, device, scaler, noise_levels, config_obj, logger, model_name="Model", metrics_dir="."):
    """评估模型在不同噪声水平下的鲁棒性"""
    logger.info(f"\n--- Evaluating Robustness for {model_name} ---")
    robustness_results = {}

    # 获取原始的、干净的测试集预测和真实值 (用于比较)
    # predict 函数现在返回 (true_values, predictions)
    original_trues_scaled, original_preds_scaled = predict(model, dataloader, device, config_obj) # 传递 config_obj

    # ... (inside evaluate_robustness function) ...

    for noise_level in noise_levels:
        logger.info(f"  Testing with noise level (std ratio): {noise_level}")
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
            logger.error(f"Error during inverse transform (noisy data): {e}. Using scaled.")
            predictions_original_noisy = pred_reshaped_noisy
            true_values_original_noisy = true_reshaped_noisy

        # Reshape back if needed for consistency, still numpy arrays on CPU
        predictions_original_noisy = predictions_original_noisy.reshape(n_samples, horizon, n_features)
        true_values_original_noisy = true_values_original_noisy.reshape(n_samples, horizon, n_features)


        # --- Calculate Metrics (on CPU using numpy) ---
        metrics_noisy = calculate_metrics(true_values_original_noisy, predictions_original_noisy, config_obj.METRICS)
        robustness_results[f"noise_{noise_level}"] = metrics_noisy
        logger.info(f"  Metrics at noise={noise_level}: {metrics_noisy}")

    # --- DataFrame creation and saving (on CPU) ---
    df_robustness = pd.DataFrame(robustness_results).T # 转置使噪声水平为行
    df_robustness.index.name = 'Noise Level (std ratio)'
    save_path = os.path.join(metrics_dir, f"{model_name}_robustness.csv")
    df_robustness.to_csv(save_path)
    logger.info(f"Robustness results saved to {save_path}")

    return df_robustness


# 稳定性评估需要在 main 脚本中通过多次运行实现
