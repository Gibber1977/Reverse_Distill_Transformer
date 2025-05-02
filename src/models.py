import torch
import torch.nn as nn
from neuralforecast.models import (
    DLinear, PatchTST, NLinear,
    Autoformer, Informer, FEDformer # Import new models from neuralforecast
)
from src import config

# Attempt to import statsmodels, but don't make it a hard requirement
# if the user only uses neural network models.
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.vector_ar.var_model import VAR
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not found. ARIMA and VAR models will not be available.")

def get_model(model_name, model_config, is_teacher=False):
    """根据名称和配置获取模型实例"""
    print(f"Initializing {'Teacher' if is_teacher else 'Student'} model: {model_name}")
    print(f"Model Config: {model_config}")

    # 确保配置中的关键参数与全局配置一致
    # 这些参数将在下面的 if/elif 块中根据模型类型添加到 cfg 副本中
    # model_config['input_size'] = config.LOOKBACK_WINDOW # Removed
    # model_config['h'] = config.PREDICTION_HORIZON       # Removed
    # model_config['n_series'] = len(config.TARGET_COLS)  # Removed

    lookback = config.LOOKBACK_WINDOW
    horizon = config.PREDICTION_HORIZON
    n_series = len(config.TARGET_COLS)

    # Copy config to avoid modifying the original dict in config.py
    cfg = model_config.copy()
    
    if model_name in ['DLinear', 'PatchTST', 'NLinear', 'Autoformer', 'Informer', 'FEDformer']:
        cfg['input_size'] = lookback
        cfg['h'] = horizon
        cfg['n_series'] = n_series
    elif model_name in ['MLP']:
        cfg['input_size'] = lookback
        cfg['h'] = horizon
        cfg['n_series'] = n_series
    elif model_name in ['RNN', 'LSTM']:
        cfg['lookback'] = lookback
        cfg['h'] = horizon
        cfg['n_series'] = n_series # Used as RNN/LSTM input_size (features)
        cfg['output_size'] = n_series # Output features per step
    
    print(f"Model Config (after applying global settings): {cfg}")

    # --- 可以添加其他模型的选择 ---
    # elif model_name == 'NLinear':
    #     model = NLinear(**model_config)
    # elif model_name == 'LSTM': # 需要自定义实现或包装
    #     # model = YourLSTMImplementation(...)
    #     raise NotImplementedError("LSTM model not implemented yet")
    # --- Model Instantiation ---
    model = None
    if model_name == 'DLinear':
        model = DLinear(**cfg)
    elif model_name == 'PatchTST':
        model = PatchTST(**cfg)
    elif model_name == 'NLinear':
        model = NLinear(**cfg)
    elif model_name == 'MLP':
        # Use custom MLP implementation
        model = MLPModel(**cfg)
    elif model_name == 'RNN':
        # Use custom RNN implementation
        model = RNNModel(**cfg)
    elif model_name == 'LSTM':
        # Use custom LSTM implementation
        model = LSTMModel(**cfg)
    elif model_name == 'Autoformer':
        model = Autoformer(**cfg)
    elif model_name == 'Informer':
        model = Informer(**cfg)
    elif model_name == 'FEDformer':
        model = FEDformer(**cfg)
    elif model_name == 'ARIMA':
        if not _STATSMODELS_AVAILABLE:
             raise ImportError("statsmodels library is required for ARIMA but not installed.")
        # ARIMA/VAR are classical models, not nn.Modules trained via backprop.
        # They don't fit naturally in this factory/training loop.
        # You would typically fit them differently (e.g., model.fit(train_data))
        # and predict (model.predict() or model.forecast()).
        raise NotImplementedError(
            f"{model_name} is a classical model (statsmodels) and doesn't fit the "
            f"neural network training paradigm expected by this factory function. "
            f"Consider handling {model_name} in a separate script or workflow."
        )
    elif model_name == 'VAR':
        if not _STATSMODELS_AVAILABLE:
             raise ImportError("statsmodels library is required for VAR but not installed.")
        if n_series <= 1:
            raise ValueError("VAR model requires multiple time series (n_series > 1).")
        # Same limitations as ARIMA apply.
        raise NotImplementedError(
            f"{model_name} is a classical model (statsmodels) and doesn't fit the "
            f"neural network training paradigm expected by this factory function. "
            f"Consider handling {model_name} in a separate script or workflow, especially for multivariate series."
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    # --- Parameter Counting (for nn.Module based models) ---
    if isinstance(model, nn.Module):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name} model initialized with {num_params:,} trainable parameters.")
    else:
        # Should not happen for implemented models, but as a fallback
        print(f"{model_name} model initialized (non-nn.Module or parameter counting failed).")
    return model

def get_teacher_model(cfg):
    """获取教师模型实例"""
    return get_model(cfg.TEACHER_MODEL_NAME, cfg.TEACHER_CONFIG.copy(), is_teacher=True)

def get_student_model(cfg):
    """获取学生模型实例"""
    return get_model(cfg.STUDENT_MODEL_NAME, cfg.STUDENT_CONFIG.copy(), is_teacher=False)

# --- 可以在这里添加自定义模型实现 ---
# class YourCustomModel(nn.Module):
#     def __init__(self, ...):
#         super().__init__()
#         # ... 定义层 ...
#     def forward(self, x):
#         # x shape: [batch_size, lookback_window, features]
#         # ... 实现前向传播 ...
#         # output shape: [batch_size, prediction_horizon, features]
#         return output

# --- 自定义模型 ---
# --- Custom Model Implementations ---
class MLPModel(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for time series forecasting."""
    def __init__(self, input_size, h, n_series, hidden_size=512, num_layers=2, activation='relu', dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.n_series = n_series
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        layers = []
        # Input layer: flatten lookback window * features
        layers.append(nn.Linear(input_size * n_series, hidden_size))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout))
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        # Output layer: predict h steps * n_series features
        layers.append(nn.Linear(hidden_size, h * n_series))
        self.model = nn.Sequential(*layers)
    def forward(self, input_dict):
        # Extract input tensor from dictionary
        x = input_dict['insample_y'] # Shape: [batch_size, lookback_window, n_series]
        batch_size = x.shape[0]
        # Flatten input: [batch_size, lookback_window * n_series]
        x = x.reshape(batch_size, -1)
        # Pass through MLP
        out = self.model(x) # Output shape: [batch_size, h * n_series]
        # Reshape output: [batch_size, h, n_series]
        out = out.reshape(batch_size, self.h, self.n_series)
        return out
class RNNModel(nn.Module):
    """A simple RNN model for time series forecasting."""
    def __init__(self, n_series, lookback, h, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.h = h
        self.output_size = output_size # Should be n_series usually
        self.n_series = n_series # Input features per time step
        self.rnn = nn.RNN(
            input_size=n_series,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Expect input as [batch, seq, feature]
            dropout=dropout if num_layers > 1 else 0 # Dropout only between layers
        )
        # Linear layer to map final hidden state to prediction horizon
        # Output needs to be h * output_size features
        self.fc = nn.Linear(hidden_size, h * output_size)
    def forward(self, input_dict):
        # Extract input tensor from dictionary
        x = input_dict['insample_y'] # Shape: [batch_size, lookback_window, n_series]
        # nn.RNN expects [batch, seq_len, input_size] which matches if batch_first=True
        batch_size = x.shape[0]
        # Pass through RNN
        # output contains hidden state for each time step: [batch, seq_len, hidden_size]
        # hidden contains the final hidden state: [num_layers, batch, hidden_size]
        output, hidden = self.rnn(x)
        # We typically use the final hidden state of the last layer
        # hidden is [num_layers, batch, hidden_size], take the last layer [-1]
        last_hidden = hidden[-1] # Shape: [batch, hidden_size]
        # Pass the final hidden state through the fully connected layer
        out = self.fc(last_hidden) # Shape: [batch_size, h * output_size]
        # Reshape output: [batch_size, h, output_size]
        out = out.view(batch_size, self.h, self.output_size)
        return out
class LSTMModel(nn.Module):
    """A simple LSTM model for time series forecasting."""
    def __init__(self, n_series, lookback, h, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.h = h
        self.output_size = output_size # Should be n_series usually
        self.n_series = n_series # Input features per time step
        self.lstm = nn.LSTM(
            input_size=n_series,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Expect input as [batch, seq, feature]
            dropout=dropout if num_layers > 1 else 0 # Dropout only between layers
        )
        # Linear layer to map final hidden state to prediction horizon
        self.fc = nn.Linear(hidden_size, h * output_size)
    def forward(self, input_dict):
        # Extract input tensor from dictionary
        x = input_dict['insample_y'] # Shape: [batch_size, lookback_window, n_series]
        # nn.LSTM expects [batch, seq_len, input_size] which matches if batch_first=True
        batch_size = x.shape[0]
        # Pass through LSTM
        # output: [batch, seq_len, hidden_size]
        # hidden: (h_n, c_n)
        # h_n: final hidden state [num_layers, batch, hidden_size]
        # c_n: final cell state [num_layers, batch, hidden_size]
        output, (hidden, cell) = self.lstm(x)
        # Use the final hidden state of the last layer
        last_hidden = hidden[-1] # Shape: [batch, hidden_size]
        # Pass through fully connected layer
        out = self.fc(last_hidden) # Shape: [batch_size, h * output_size]
        # Reshape output: [batch_size, h, output_size]
        out = out.view(batch_size, self.h, self.output_size)
        return out