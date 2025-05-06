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

    # 全局配置
    lookback = config.LOOKBACK_WINDOW
    horizon = config.PREDICTION_HORIZON
    n_series = len(config.TARGET_COLS)

    # 使用原始 model_config，避免修改
    cfg = model_config # Use the original config passed

    print(f"Using Base Config: {cfg}")
    print(f"Global Params: lookback={lookback}, horizon={horizon}, n_series={n_series}")

    # --- 可以添加其他模型的选择 ---
    # elif model_name == 'NLinear':
    #     model = NLinear(**model_config)
    # elif model_name == 'LSTM': # 需要自定义实现或包装
    #     # model = YourLSTMImplementation(...)
    #     raise NotImplementedError("LSTM model not implemented yet")
    # --- Model Instantiation ---
    model = None
    if model_name == 'DLinear':
        # NeuralForecast models often expect input_size, h, n_series
        # Ensure cfg has the correct input_size, h, n_series (handled in main.py update_config)
        model = DLinear(**cfg)
    elif model_name == 'PatchTST':
        # Ensure cfg has the correct input_size, h, n_series
        model = PatchTST(**cfg)
    elif model_name == 'NLinear':
        # Ensure cfg has the correct input_size, h, n_series
        model = NLinear(**cfg)
    elif model_name == 'MLP':
        # Custom MLP expects specific args from its __init__
        model = MLPModel(input_size=lookback, h=horizon, n_series=n_series,
                         hidden_size=cfg.get('hidden_size', 512),
                         num_layers=cfg.get('num_layers', 2),
                         activation=cfg.get('activation', 'relu'),
                         dropout=cfg.get('dropout', 0.1))
    elif model_name == 'RNN':
        # Custom RNN expects specific args
        model = RNNModel(n_series=n_series, lookback=lookback, h=horizon,
                         hidden_size=cfg.get('hidden_size', 128),
                         num_layers=cfg.get('num_layers', 2),
                         dropout=cfg.get('dropout', 0.1),
                         output_size=n_series) # Assuming output size matches n_series
    elif model_name == 'LSTM':
        # Custom LSTM expects specific args
        model = LSTMModel(n_series=n_series, lookback=lookback, h=horizon,
                          hidden_size=cfg.get('hidden_size', 128),
                          num_layers=cfg.get('num_layers', 2),
                          dropout=cfg.get('dropout', 0.1),
                          output_size=n_series) # Assuming output size matches n_series
    elif model_name == 'Autoformer':
        # Ensure cfg has the correct input_size, h, n_series
        model = Autoformer(**cfg)
    elif model_name == 'Informer':
        # Ensure cfg has the correct input_size, h, n_series
        model = Informer(**cfg)
    elif model_name == 'FEDformer':
        # Ensure cfg has the correct input_size, h, n_series
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