import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from src import config
import os

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

def load_and_preprocess_data(cfg):
    """加载、预处理、划分和创建 DataLoaders"""
    print("--- Starting Data Loading and Preprocessing ---")

    # --- 1. 加载数据 ---
    try:
        df = pd.read_csv(cfg.DATASET_PATH)
        print(f"Dataset loaded successfully from {cfg.DATASET_PATH}")
        print(f"Original data shape: {df.shape}")
        print("Original data columns:", df.columns.tolist())
        print("Original data types:\n", df.dtypes)
        print("Original data head:\n", df.head())
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {cfg.DATASET_PATH}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
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
         print(f"Error: Date column '{cfg.DATE_COL}' not found in the dataset.")
         raise
    except Exception as e:
         print(f"Error processing date column: {e}. Trying without setting index.")
         # 如果日期处理复杂或失败，可以尝试不设置索引，但后续可能需要手动排序
         df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], utc=True, errors='coerce')
         df = df.sort_values(by=cfg.DATE_COL).dropna(subset=[cfg.DATE_COL])


    # 选择目标列
    try:
        df_target = df[cfg.TARGET_COLS]
    except KeyError:
        print(f"Error: One or more target columns {cfg.TARGET_COLS} not found.")
        raise

    # 处理缺失值 (简单填充，可以用更复杂的方法)
    if df_target.isnull().values.any():
        print(f"Warning: Missing values found in target columns. Filling with forward fill.")
        df_target = df_target.ffill().bfill() # 先前向填充，再后向填充处理开头的 NaN
    if df_target.isnull().values.any():
        print(f"Error: Still missing values after fill. Check data.")
        # 可以选择填充为 0 或均值，但可能引入偏差
        # df_target = df_target.fillna(0)
        raise ValueError("Data contains NaNs after attempting fill.")

    data_values = df_target.values.astype(np.float32)
    print(f"Target data shape: {data_values.shape}")

    # --- 3. 划分数据 ---
    n_total = len(data_values)
    n_test = int(n_total * cfg.TEST_SPLIT_RATIO)
    n_train_val = n_total - n_test
    n_val = int(n_train_val * cfg.VAL_SPLIT_RATIO)
    n_train = n_train_val - n_val

    if n_train < cfg.LOOKBACK_WINDOW + cfg.PREDICTION_HORIZON or \
       n_val < cfg.LOOKBACK_WINDOW + cfg.PREDICTION_HORIZON or \
       n_test < cfg.LOOKBACK_WINDOW + cfg.PREDICTION_HORIZON:
        print("Warning: Not enough data for the specified lookback, horizon, and splits.")
        print(f"n_train={n_train}, n_val={n_val}, n_test={n_test}")
        # 可以考虑减少划分比例或调整窗口大小

    train_data = data_values[:n_train]
    val_data = data_values[n_train : n_train + n_val]
    test_data = data_values[n_train + n_val :]

    print(f"Data split: Train={train_data.shape[0]}, Validation={val_data.shape[0]}, Test={test_data.shape[0]}")

    # --- 4. 数据标准化 ---
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)
    print("Data scaled using StandardScaler (fit on train set).")

    # --- 5. 创建 Datasets 和 DataLoaders ---
    train_dataset = TimeSeriesDataset(train_data_scaled, cfg.LOOKBACK_WINDOW, cfg.PREDICTION_HORIZON)
    val_dataset = TimeSeriesDataset(val_data_scaled, cfg.LOOKBACK_WINDOW, cfg.PREDICTION_HORIZON)
    test_dataset = TimeSeriesDataset(test_data_scaled, cfg.LOOKBACK_WINDOW, cfg.PREDICTION_HORIZON)

    # 注意：时间序列通常不在训练时 shuffle，以利用样本间的时序关系
    # 使用自定义的 collate_fn 来处理 Dataset 返回的 None 值
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False, collate_fn=collate_fn_skip_none)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False, collate_fn=collate_fn_skip_none)

    print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    print("--- Data Loading and Preprocessing Finished ---")

    return train_loader, val_loader, test_loader, scaler # 返回 scaler 用于逆变换
