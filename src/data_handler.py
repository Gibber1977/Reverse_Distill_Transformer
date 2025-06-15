import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from src import config
import os
import random
import math

def time_features(dates, freq='h'):
    """
    Generates time features from a pandas DatetimeIndex.
    Args:
        dates (pd.DatetimeIndex): Datetime index to extract features from.
        freq (str): Frequency of the time series ('h' for hour, 'd' for day, 'w' for week, 'm' for month, 'q' for quarter, 'y' for year, 'min' for minute).
    Returns:
        np.ndarray: Array of time features.
    """
    features = []
    if freq == 'min': # 支持分钟级
        features.append(dates.dt.minute.values / 59.0 - 0.5) # Normalize to [-0.5, 0.5]
        features.append(dates.dt.hour.values / 23.0 - 0.5)
        features.append(dates.dt.dayofweek.values / 6.0 - 0.5)
        features.append(dates.dt.dayofyear.values / 365.0 - 0.5)
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.isocalendar().week.values / 52.0 - 0.5)
    elif freq == 'h':
        features.append(dates.dt.hour.values / 23.0 - 0.5)
        features.append(dates.dt.dayofweek.values / 6.0 - 0.5)
        features.append(dates.dt.dayofyear.values / 365.0 - 0.5)
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.isocalendar().week.values / 52.0 - 0.5)
    elif freq == 'd':
        features.append(dates.dt.dayofweek.values / 6.0 - 0.5)
        features.append(dates.dt.dayofyear.values / 365.0 - 0.5)
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.isocalendar().week.values / 52.0 - 0.5)
    elif freq == 'w':
        features.append(dates.dt.dayofweek.values / 6.0 - 0.5)
        features.append(dates.dt.dayofyear.values / 365.0 - 0.5)
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.isocalendar().week.values / 52.0 - 0.5)
    elif freq == 'm':
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.quarter.values / 3.0 - 0.5)
        features.append(dates.dt.year.values)
    elif freq == 'q':
        features.append(dates.dt.quarter.values / 3.0 - 0.5)
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.year.values)
    elif freq == 'y':
        features.append(dates.dt.year.values)
    else: # Default to hour, dayofweek, dayofyear, month, weekofyear (same as 'h')
        features.append(dates.dt.hour.values / 23.0 - 0.5)
        features.append(dates.dt.dayofweek.values / 6.0 - 0.5)
        features.append(dates.dt.dayofyear.values / 365.0 - 0.5)
        features.append(dates.dt.month.values / 11.0 - 0.5)
        features.append(dates.dt.isocalendar().week.values / 52.0 - 0.5)

    return np.array(features).transpose(1, 0)

def cyclic_time_features(dates, freq='h'):
    """
    Generates cyclic time features (sin/cos) from a pandas DatetimeIndex.
    Args:
        dates (pd.DatetimeIndex): Datetime index to extract features from.
        freq (str): Frequency of the time series ('h' for hour, 'd' for day, etc., 'min' for minute).
    Returns:
        np.ndarray: Array of cyclic time features.
    """
    features = []
    if freq == 'min': # 支持分钟级
        # Minute (0-59)
        features.append(np.sin(2 * math.pi * dates.dt.minute.values / 60.0))
        features.append(np.cos(2 * math.pi * dates.dt.minute.values / 60.0))
        # Hour (0-23)
        features.append(np.sin(2 * math.pi * dates.dt.hour.values / 24.0))
        features.append(np.cos(2 * math.pi * dates.dt.hour.values / 24.0))
        # Day of week (0-6)
        features.append(np.sin(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        # Day of year (1-366)
        features.append(np.sin(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        # Month (1-12)
        features.append(np.sin(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.cos(2 * math.pi * dates.dt.month.values / 12.0))
        # Week of year (1-53)
        features.append(np.sin(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
        features.append(np.cos(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
    elif freq == 'h':
        # Hour (0-23)
        features.append(np.sin(2 * math.pi * dates.dt.hour.values / 24.0))
        features.append(np.cos(2 * math.pi * dates.dt.hour.values / 24.0))
        # Day of week (0-6)
        features.append(np.sin(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        # Day of year (1-366)
        features.append(np.sin(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        # Month (1-12)
        features.append(np.sin(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.cos(2 * math.pi * dates.dt.month.values / 12.0))
        # Week of year (1-53)
        features.append(np.sin(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
        features.append(np.cos(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
    elif freq == 'd':
        # Day of week (0-6)
        features.append(np.sin(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        # Day of year (1-366)
        features.append(np.sin(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        # Month (1-12)
        features.append(np.sin(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.cos(2 * math.pi * dates.dt.month.values / 12.0))
        # Week of year (1-53)
        features.append(np.sin(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
        features.append(np.cos(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
    elif freq == 'w':
        # Day of week (0-6) - if weekly data, this might not be relevant or always 0
        features.append(np.sin(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        # Month (1-12)
        features.append(np.sin(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.cos(2 * math.pi * dates.dt.month.values / 12.0))
        # Week of year (1-53)
        features.append(np.sin(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
        features.append(np.cos(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
    elif freq == 'm':
        # Month (1-12)
        features.append(np.sin(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.cos(2 * math.pi * dates.dt.month.values / 12.0))
        # Quarter (1-4)
        features.append(np.sin(2 * math.pi * dates.dt.quarter.values / 4.0))
        features.append(np.cos(2 * math.pi * dates.dt.quarter.values / 4.0))
    elif freq == 'q':
        # Quarter (1-4)
        features.append(np.sin(2 * math.pi * dates.dt.quarter.values / 4.0))
        features.append(np.cos(2 * math.pi * dates.dt.quarter.values / 4.0))
        # Month (1-12)
        features.append(np.sin(2 * math.pi * dates.month.values / 12.0)) # Original code had dates.month, should be dates.dt.month
        features.append(np.cos(2 * math.pi * dates.month.values / 12.0)) # Original code had dates.month, should be dates.dt.month
    elif freq == 'y':
        # Year is not typically cyclic in this sense, but if needed, could use a very long cycle or specific events
        # For now, no cyclic features for year itself.
        pass
    else: # Default to hour, dayofweek, dayofyear, month, weekofyear (same as 'h')
        features.append(np.sin(2 * math.pi * dates.dt.hour.values / 24.0))
        features.append(np.cos(2 * math.pi * dates.dt.hour.values / 24.0))
        features.append(np.sin(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofweek.values / 7.0))
        features.append(np.sin(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        features.append(np.cos(2 * math.pi * dates.dt.dayofyear.values / 366.0))
        features.append(np.sin(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.cos(2 * math.pi * dates.dt.month.values / 12.0))
        features.append(np.sin(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))
        features.append(np.cos(2 * math.pi * dates.dt.isocalendar().week.values / 53.0))

    return np.array(features).transpose(1, 0)

def add_noise(data, noise_type, noise_level):
    """
    向数据添加指定类型的噪声。
    data: numpy array, 原始数据
    noise_type: 字符串, 噪声类型 ('gaussian', 'salt_and_pepper', 'poisson')
    noise_level: 浮点数, 噪声水平 (高斯噪声的标准差比例, 椒盐噪声的比例, 泊松噪声的强度)
    """
    if noise_type == 'gaussian':
        # 高斯噪声: 均值为0，标准差为数据标准差的 noise_level 倍
        # 确保 noise_level 是一个合理的比例，例如 0.01 到 0.1
        std_dev = np.std(data)
        noise = np.random.normal(0, std_dev * noise_level, data.shape)
        noisy_data = data + noise
    elif noise_type == 'salt_and_pepper':
        # 椒盐噪声: 随机将数据点设为最小值或最大值
        noisy_data = np.copy(data)
        num_salt = np.ceil(noise_level * data.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in data.shape]
        noisy_data[tuple(coords)] = data.max() # Salt

        num_pepper = np.ceil(noise_level * data.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in data.shape]
        noisy_data[tuple(coords)] = data.min() # Pepper
    elif noise_type == 'poisson':
        # 泊松噪声: 适用于计数数据，这里简单模拟
        # 假设数据是非负的，且可以被视为计数
        # 将数据缩放到一个合适的范围，然后添加泊松噪声
        # 注意：泊松噪声通常应用于整数计数，这里为浮点数据做近似
        noisy_data = np.random.poisson(data * noise_level) / noise_level # 缩放回来
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    return noisy_data

def smooth_data(data, smoothing_method="moving_average", smoothing_factor=24, 
                weight_original=None, weight_smoothing=None):
    """
    对数据进行平滑处理，并可选择与原始数据加权合成。
    
    data: numpy array, 原始数据
    smoothing_method: 字符串, 平滑方法 ('moving_average', 'exponential', 'none')
    smoothing_factor: 浮点数, 平滑系数 (例如，移动平均的窗口大小或指数平滑的alpha)
    weight_original: 浮点数, 原始数据在合成中的权重 (0.0 到 1.0)。
                     如果为 None 且 weight_smoothing 也为 None，则默认为 0.8。
    weight_smoothing: 浮点数, 平滑数据在合成中的权重 (0.0 到 1.0)。
                      如果为 None 且 weight_original 也为 None，则默认为 0.2。
                      weight_original 和 weight_smoothing 必须和为 1.0。
                      两者只能传入一个，或都不传入。
    """
    if smoothing_method == 'moving_average':
        if not isinstance(smoothing_factor, int) or smoothing_factor <= 0:
            raise ValueError("For 'moving_average', smoothing_factor must be a positive integer (window size).")
        window_size = int(smoothing_factor)
        temp_smoothed_data = np.copy(data).astype(float) # 确保数据类型为浮点，避免整数截断
        for i in range(data.shape[1]): # 对每个特征列进行平滑
            temp_smoothed_data[:, i] = pd.Series(data[:, i]).rolling(
                window=window_size, min_periods=1, center=True
            ).mean().values
    elif smoothing_method == 'exponential':
        if not (0 <= smoothing_factor <= 1):
            raise ValueError("For 'exponential', smoothing_factor must be between 0 and 1 (alpha).")
        alpha = smoothing_factor
        temp_smoothed_data = np.copy(data).astype(float) # 确保数据类型为浮点
        for i in range(data.shape[1]): # 对每个特征列进行平滑
            temp_smoothed_data[:, i] = pd.Series(data[:, i]).ewm(alpha=alpha, adjust=False).mean().values
    elif smoothing_method == 'none':
        temp_smoothed_data = np.copy(data).astype(float) # 确保数据类型为浮点
    else:
        raise ValueError(f"Unsupported smoothing method: {smoothing_method}")
    
    # --- 处理权重参数 ---
    if weight_original is not None and weight_smoothing is not None:
        raise ValueError("Please provide either 'weight_original' or 'weight_smoothing', not both.")
    
    if weight_original is None and weight_smoothing is None:
        # 都没有传入，使用默认值
        weight_original = 0.8
        weight_smoothing = 0.2
    elif weight_original is not None:
        # 只传入了 weight_original
        if not (0.0 <= weight_original <= 1.0):
            raise ValueError("weight_original must be between 0.0 and 1.0.")
        weight_smoothing = 1.0 - weight_original
    elif weight_smoothing is not None:
        # 只传入了 weight_smoothing
        if not (0.0 <= weight_smoothing <= 1.0):
            raise ValueError("weight_smoothing must be between 0.0 and 1.0.")
        weight_original = 1.0 - weight_smoothing
    
    # 确保最终权重和为 1 (理论上到这里已经是了，但明确一下)
    # 浮点数比较需要注意精度，但这里是直接计算得出的，所以通常没问题
    if not np.isclose(weight_original + weight_smoothing, 1.0):
        # 理论上不会发生，除非上述逻辑有bug
        raise RuntimeError("Internal error: weight_original and weight_smoothing do not sum to 1.0 after calculation.")
    # 加权合成平滑后的数据和原始数据
    smoothed_data = weight_smoothing * temp_smoothed_data + weight_original * data
    
    return smoothed_data

class TimeSeriesDataset(Dataset):
    """用于时间序列数据的 PyTorch Dataset"""
    def __init__(self, data, lookback, horizon):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = data.shape[1]

        # 生成样本索引
        self.indices = []
        for i in range(len(data) - lookback - horizon + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        x_end_idx = start_idx + self.lookback
        y_start_idx = x_end_idx
        y_end_idx = y_start_idx + self.horizon

        x = self.data[start_idx:x_end_idx, :]
        y = self.data[y_start_idx:y_end_idx, :]

        # 转换为 torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # 返回 x, y 以及 None 作为外生变量占位符
        return x, y, None, None
import torch.utils.data

def collate_fn_skip_none(batch):
    """
    自定义 collate_fn，用于处理 Dataset 返回元组中可能包含 None 的情况。
    它会正常 collate 非 None 的元素（假定为 Tensors），并保留 None 值。
    """
    # batch 是一个列表，每个元素是 Dataset.__getitem__ 的返回值 (x, y, hist_exog, futr_exog)
    # 例如: [(x1, y1, None, None), (x2, y2, None, None), ...]

    # 分离不同类型的元素
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    hist_exogs = [item[2] for item in batch] # 这将是 [None, None, ...]
    futr_exogs = [item[3] for item in batch] # 这将是 [None, None, ...]

    # Collate Tensors
    collated_xs = torch.stack(xs, 0)
    collated_ys = torch.stack(ys, 0)

    # 处理外生变量 - 如果所有都是 None，则返回 None；否则尝试 collate（如果将来支持）
    # 在当前情况下，它们总是 None
    collated_hist_exogs = None if all(x is None for x in hist_exogs) else torch.utils.data.default_collate(hist_exogs) # 或者更复杂的处理
    collated_futr_exogs = None if all(x is None for x in futr_exogs) else torch.utils.data.default_collate(futr_exogs) # 或者更复杂的处理

    return collated_xs, collated_ys, collated_hist_exogs, collated_futr_exogs

def load_and_preprocess_data(dataset_path, cfg, logger, time_freq):
    """加载、预处理、划分和创建 DataLoaders"""
    logger.info("--- Starting Data Loading and Preprocessing ---")

    # --- 1. 加载数据 ---
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded successfully from {dataset_path}")
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Original data columns: {df.columns.tolist()}")
        logger.info(f"Original data types:\n{df.dtypes}")
        logger.info(f"Original data head:\n{df.head()}")
    except FileNotFoundError:
        logger.error(f"Error: Dataset file not found at {dataset_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # --- 2. 预处理 ---
    # 解析日期列并设为索引
    try:
        df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], utc=True) # 解析并设为 UTC
        # df = df.set_index(cfg.DATE_COL).sort_index() # 设为索引并排序
        # 检查频率是否一致 (如果需要)
        # inferred_freq = pd.infer_freq(df.index)
        # print(f"Inferred frequency: {inferred_freq}")
        # if inferred_freq != cfg.TIME_FREQ:
        #     print(f"Warning: Inferred frequency '{inferred_freq}' does not match configured frequency '{cfg.TIME_FREQ}'. Resampling...")
        #     df = df.resample(cfg.TIME_FREQ).mean() # 或 .median(), .first() 等，根据需要填充缺失
    except KeyError:
         logger.error(f"Error: Date column '{cfg.DATE_COL}' not found in the dataset.")
         raise
    except Exception as e:
         logger.error(f"Error processing date column: {e}. Trying without setting index.")
         # 如果日期处理复杂或失败，可以尝试不设置索引，但后续可能需要手动排序
         df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], utc=True, errors='coerce')
         df = df.sort_values(by=cfg.DATE_COL).dropna(subset=[cfg.DATE_COL])


    # 选择目标列
    try:
        df_target = df[cfg.TARGET_COLS]
    except KeyError:
        logger.error(f"Error: One or more target columns {cfg.TARGET_COLS} not found.")
        raise

    # --- 自动识别和合并协变量 ---
    # 1. 识别所有非目标、非日期的列作为原始协变量
    original_exog_cols = [col for col in df.columns if col not in cfg.TARGET_COLS and col != cfg.DATE_COL]
    df_original_exog = df[original_exog_cols]
    logger.info(f"Automatically identified original exogenous columns: {original_exog_cols}")

    # 2. 处理在 config 中明确指定的 EXOGENOUS_COLS
    df_specified_exog = pd.DataFrame(index=df.index)
    if hasattr(cfg, 'EXOGENOUS_COLS') and cfg.EXOGENOUS_COLS:
        existing_specified_cols = [col for col in cfg.EXOGENOUS_COLS if col in df.columns]
        if existing_specified_cols:
            df_specified_exog = df[existing_specified_cols]
            logger.info(f"Selected specified exogenous columns from config: {existing_specified_cols}")
        else:
            logger.warning(f"None of the specified EXOGENOUS_COLS {cfg.EXOGENOUS_COLS} found in the dataset.")
    
    # 3. 合并所有协变量 (原始 + 指定的)，去重
    df_exog = pd.concat([df_original_exog, df_specified_exog], axis=1)
    df_exog = df_exog.loc[:,~df_exog.columns.duplicated()] # 去除重复的列
    logger.info(f"Total exogenous columns to be used (before time features): {df_exog.columns.tolist()}")


    # --- 时间特征编码 ---
    dates = df[cfg.DATE_COL]
    time_features_data = None
    if cfg.TIME_ENCODING_TYPE in ['linear', 'cyclic']:
        logger.info(f"Applying {cfg.TIME_ENCODING_TYPE} time encoding with frequency: {time_freq}")
        if cfg.TIME_ENCODING_TYPE == 'linear':
            time_features_data = time_features(dates, freq=time_freq)
        else:
            time_features_data = cyclic_time_features(dates, freq=time_freq)
    else:
        logger.warning(f"Unknown or no TIME_ENCODING_TYPE specified: '{cfg.TIME_ENCODING_TYPE}'. No time features will be added.")

    # --- 合并所有特征 ---
    # 确保目标列始终在最前面
    dfs_to_concat = [df_target]
    
    # 添加所有协变量
    if not df_exog.empty:
        dfs_to_concat.append(df_exog)
        
    # 添加时间特征
    if time_features_data is not None:
        time_feature_cols = [f'time_feature_{i}' for i in range(time_features_data.shape[1])]
        df_time_features = pd.DataFrame(time_features_data, index=df.index, columns=time_feature_cols)
        dfs_to_concat.append(df_time_features)
    
    df_processed = pd.concat(dfs_to_concat, axis=1)
    logger.info(f"Data processed. Final data shape: {df_processed.shape}")
    logger.info(f"Final data columns: {df_processed.columns.tolist()}")


    # 处理缺失值 (简单填充，可以用更复杂的方法)
    if df_processed.isnull().values.any():
        logger.warning(f"Missing values found in processed data. Filling with forward fill.")
        df_processed = df_processed.ffill().bfill() # 先前向填充，再后向填充处理开头的 NaN
    if df_processed.isnull().values.any():
        logger.error(f"Still missing values after fill in processed data. Check data.")
        # 可以选择填充为 0 或均值，但可能引入偏差
        # df_processed = df_processed.fillna(0)
        raise ValueError("Processed data contains NaNs after attempting fill.")

    # --- 3. 划分数据 ---
    n_total = len(df_processed)
    n_test = int(n_total * cfg.TEST_SPLIT_RATIO)
    n_train_val = n_total - n_test
    n_val = int(n_train_val * cfg.VAL_SPLIT_RATIO)
    n_train = n_train_val - n_val

    if n_train < cfg.LOOKBACK_WINDOW + cfg.PREDICTION_HORIZON or \
       n_val < cfg.LOOKBACK_WINDOW + cfg.PREDICTION_HORIZON or \
       n_test < cfg.LOOKBACK_WINDOW + cfg.PREDICTION_HORIZON:
        logger.warning("Not enough data for the specified lookback, horizon, and splits.")
        logger.warning(f"n_train={n_train}, n_val={n_val}, n_test={n_test}")
        # 可以考虑减少划分比例或调整窗口大小

    train_data_raw = df_processed.iloc[:n_train].values.astype(np.float32)
    val_data_raw = df_processed.iloc[n_train : n_train + n_val].values.astype(np.float32)
    test_data_raw = df_processed.iloc[n_train + n_val :].values.astype(np.float32)

    logger.info(f"Raw Data split: Train={train_data_raw.shape[0]}, Validation={val_data_raw.shape[0]}, Test={test_data_raw.shape[0]}")

    # --- 4. 数据平滑 (可选) ---
    # 对原始数据进行平滑，然后标准化
    if cfg.SMOOTHING_APPLY_TRAIN and cfg.SMOOTHING_METHOD != 'none':
        logger.info(f"Applying smoothing to training data: method='{cfg.SMOOTHING_METHOD}', window_size={cfg.SMOOTHING_WINDOW_SIZE}, weight_smoothing={cfg.WEIGHT_SMOOTHING}")
        train_data_smoothed = smooth_data(train_data_raw, smoothing_method=cfg.SMOOTHING_METHOD, smoothing_factor=cfg.SMOOTHING_WINDOW_SIZE, weight_smoothing=cfg.WEIGHT_SMOOTHING)
    else:
        train_data_smoothed = train_data_raw
    
    if cfg.SMOOTHING_APPLY_VAL and cfg.SMOOTHING_METHOD != 'none':
        logger.info(f"Applying smoothing to validation data: method='{cfg.SMOOTHING_METHOD}', window_size={cfg.SMOOTHING_WINDOW_SIZE}, weight_smoothing={cfg.WEIGHT_SMOOTHING}")
        val_data_smoothed = smooth_data(val_data_raw, smoothing_method=cfg.SMOOTHING_METHOD, smoothing_factor=cfg.SMOOTHING_WINDOW_SIZE, weight_smoothing=cfg.WEIGHT_SMOOTHING)
    else:
        val_data_smoothed = val_data_raw

    # 测试集通常不进行平滑，除非是去噪评估的一部分
    if cfg.SMOOTHING_APPLY_TEST and cfg.SMOOTHING_METHOD != 'none':
        logger.info(f"Applying smoothing to test data: method='{cfg.SMOOTHING_METHOD}', window_size={cfg.SMOOTHING_WINDOW_SIZE}, weight_smoothing={cfg.WEIGHT_SMOOTHING}")
        test_data_smoothed = smooth_data(test_data_raw, smoothing_method=cfg.SMOOTHING_METHOD, smoothing_factor=cfg.SMOOTHING_WINDOW_SIZE, weight_smoothing=cfg.WEIGHT_SMOOTHING)
    else:
        test_data_smoothed = test_data_raw

    # --- 5. 数据标准化 ---
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_smoothed)
    val_data_scaled = scaler.transform(val_data_smoothed)
    test_data_scaled = scaler.transform(test_data_smoothed) # 测试集也用训练集的 scaler 转换
    logger.info("Data scaled using StandardScaler (fit on train set).")

    # --- 6. 噪音注入 (可选) ---
    if cfg.TRAIN_NOISE_INJECTION_LEVEL > 0 and cfg.NOISE_TYPE != 'none':
        logger.info(f"Applying noise injection to training data: type='{cfg.NOISE_TYPE}', level={cfg.TRAIN_NOISE_INJECTION_LEVEL}")
        train_data_scaled = add_noise(train_data_scaled, cfg.NOISE_TYPE, cfg.TRAIN_NOISE_INJECTION_LEVEL)
        logger.info("Noise injected into training data.")
    
    if cfg.VAL_NOISE_INJECTION_LEVEL > 0 and cfg.NOISE_TYPE != 'none':
        logger.info(f"Applying noise injection to validation data: type='{cfg.NOISE_TYPE}', level={cfg.VAL_NOISE_INJECTION_LEVEL}")
        val_data_scaled = add_noise(val_data_scaled, cfg.NOISE_TYPE, cfg.VAL_NOISE_INJECTION_LEVEL)
        logger.info("Noise injected into validation data.")

    # --- 7. 创建 Datasets 和 DataLoaders ---
    train_dataset = TimeSeriesDataset(train_data_scaled, cfg.LOOKBACK_WINDOW, cfg.PREDICTION_HORIZON)
    val_dataset = TimeSeriesDataset(val_data_scaled, cfg.LOOKBACK_WINDOW, cfg.PREDICTION_HORIZON)
    test_dataset = TimeSeriesDataset(test_data_scaled, cfg.LOOKBACK_WINDOW, cfg.PREDICTION_HORIZON)

    # 注意：时间序列通常不在训练时 shuffle，以利用样本间的时序关系
    # 使用自定义的 collate_fn 来处理 Dataset 返回的 None 值
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False, collate_fn=collate_fn_skip_none,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False, collate_fn=collate_fn_skip_none,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False, collate_fn=collate_fn_skip_none,pin_memory=True)

    logger.info(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    logger.info("--- Data Loading and Preprocessing Finished ---")

    return train_loader, val_loader, test_loader, scaler, df_processed.shape[1]
