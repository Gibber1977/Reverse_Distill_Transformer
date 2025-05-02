# --- Data Preprocessing Module ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path, target_col, feature_cols, seq_len, pred_len):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)

    # Basic Cleaning & Feature Selection
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df = df.set_index('Formatted Date')
    df = df.sort_index()

    # Ensure hourly frequency (handle potential duplicates/missing indices)
    # This might take a moment on large datasets
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('H') # Resample to hourly frequency

    # Select features + target
    df_features = df[feature_cols].copy()

    # Simple missing value handling (forward fill)
    df_features.ffill(inplace=True)
    df_features.bfill(inplace=True) # Backfill remaining NaNs at the beginning

    # Data Scaling (Fit only on training data!)
    n_features = df_features.shape[1]
    scaler = StandardScaler()

    # Temporal Splitting (70-15-15)
    n = len(df_features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_data = df_features.iloc[:train_end].values
    val_data = df_features.iloc[train_end:val_end].values
    test_data = df_features.iloc[val_end:].values

    print(f"Raw data shapes: Train={train_data.shape}, Val={val_data.shape}, Test={test_data.shape}")

    # Fit scaler ONLY on training data
    scaler.fit(train_data)

    # Scale data
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Target column index
    try:
        target_col_index = feature_cols.index(target_col)
    except ValueError:
        raise ValueError(f"Target column '{target_col}' not found in feature columns: {feature_cols}")

    # Generate sequences
    X_train, Y_true_train = create_sequences(train_scaled, seq_len, pred_len, target_col_index)
    X_val, Y_true_val = create_sequences(val_scaled, seq_len, pred_len, target_col_index)
    X_test, Y_true_test = create_sequences(test_scaled, seq_len, pred_len, target_col_index)

    print(f"Sequence shapes: X_train={X_train.shape}, Y_true_train={Y_true_train.shape}")
    print(f"Sequence shapes: X_val={X_val.shape}, Y_true_val={Y_true_val.shape}")
    print(f"Sequence shapes: X_test={X_test.shape}, Y_true_test={Y_true_test.shape}")

    return X_train, Y_true_train, X_val, Y_true_val, X_test, Y_true_test, scaler, target_col_index, n_features

def create_sequences(data, seq_len, pred_len, target_idx):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len:i + seq_len + pred_len, target_idx] # Only target column
        xs.append(x)
        ys.append(y)
    # Ensure shapes are correct: X=(samples, seq_len, features), Y=(samples, pred_len)
    return np.array(xs), np.array(ys).reshape(-1, pred_len, 1) # Reshape Y to have a feature dim of 1