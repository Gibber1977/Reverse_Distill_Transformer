import torch
import os

# --- 基本配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 获取项目根目录 rdt_framework/
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
# METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
# MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
# PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# --- 数据集配置 ---
DATASET_PATH = os.path.join(DATA_DIR, 'national_illness.csv')
DATE_COL = 'date'         # 日期/时间列名
TARGET_COLS = ['OT']   # 需要预测的目标列（可以是多个）
# TARGET_COLS = ['Temperature (C)', 'Humidity'] # 示例：多变量预测
TIME_FREQ = 'D'                     # 时间序列频率 ('H' for hourly) - 需要根据你的数据调整

# --- 数据处理配置 ---
LOOKBACK_WINDOW = 96     # 回看窗口大小 (e.g., 4 days for hourly data)
PREDICTION_HORIZON = 192   # 预测未来步长 (e.g., 1 day for hourly data)
VAL_SPLIT_RATIO = 0.43    # 验证集比例 (从训练集中划分) - Increased
TEST_SPLIT_RATIO = 0.3     # 测试集比例 (从总数据末尾划分) - Increased
BATCH_SIZE = 8192          # 批次大小 (根据 GPU 内存调整) - Reduced significantly
NUM_WORKERS = 0           # DataLoader 的工作进程数 (Windows 设为 0 可能更稳定)

# --- 模型配置 ---
# 教师模型 (DLinear) - 也可以选择 NLinear, 但 DLinear 通常更稳健
TEACHER_MODEL_NAME = 'DLinear'
TEACHER_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS), # 自动根据目标列数设置
    'hidden_size': 256,
    'moving_avg_window': 25, # DLinear 的超参数
    # 'individual': False # 是否为每个序列单独建模 (如果 n_series > 1)
}

# 学生模型 (PatchTST)
STUDENT_MODEL_NAME = 'PatchTST'
STUDENT_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS), # 自动根据目标列数设置
    'patch_len': 16,        # PatchTST 的超参数
    'stride': 8,           # PatchTST 的超参数
    'n_layers': 3,
    'n_heads': 4,
    'hidden_size': 128,
    'ff_hidden_size': 256,
    'revin': True,          # 是否使用 Reversible Instance Normalization
    # 'revin_affine': True # revin 的子参数
}

# --- 训练配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500              # 最大训练轮数
LEARNING_RATE = 0.001
OPTIMIZER = 'Adam'      # 可选 'AdamW', 'SGD' 等
WEIGHT_DECAY = 1e-5     # L2 正则化 (用于 AdamW)
PATIENCE = 50            # 早停耐心轮数
LOSS_FN = 'MSE'         # 任务损失 ('MSE' 或 'MAE')

# --- RDT 配置 ---
ALPHA_START = 0.3       # RDT 初始 alpha
ALPHA_END = 0.7         # RDT 最终 alpha
ALPHA_SCHEDULE = 'linear' # 调度器类型 ('linear', 'exponential', 'constant')
CONSTANT_ALPHA = 0.5    # 如果 ALPHA_SCHEDULE = 'constant'，使用此 alpha 值

# --- 评估配置 ---
METRICS = ['mae', 'mse']
ROBUSTNESS_NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1] # 添加到测试集的高斯噪声标准差比例

STABILITY_RUNS = 1      # 运行多次以评估稳定性 (设为 1 则不进行稳定性评估)

# --- 实验管理 ---
SEED = 42               # 随机种子，用于可复现性
EXPERIMENT_NAME = f"RDT_{STUDENT_MODEL_NAME}_vs_{TEACHER_MODEL_NAME}_h{PREDICTION_HORIZON}"

# # --- 创建结果目录 ---
# os.makedirs(METRICS_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 动态更新模型配置中的 n_series ---
TEACHER_CONFIG['n_series'] = len(TARGET_COLS)
STUDENT_CONFIG['n_series'] = len(TARGET_COLS)

# --- 新增模型参数设置 ---
# NLinear (Already in neuralforecast)
NLINEAR_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS),
    # 'individual': False # Optional: True for separate models per series
}
# MLP (Custom Implementation)
MLP_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS),
    'hidden_size': 512,
    'num_layers': 2, # Number of hidden layers
    'activation': 'relu', # 'relu', 'tanh', etc.
    'dropout': 0.1
}
# RNN (Custom Implementation)
RNN_CONFIG = {
    # Note: input_size for nn.RNN is features per time step (n_series)
    'n_series': len(TARGET_COLS), # Used as input_size for nn.RNN layer
    'lookback': LOOKBACK_WINDOW, # Used to define sequence length
    'h': PREDICTION_HORIZON, # Prediction horizon
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'output_size': len(TARGET_COLS) # Final output features per step
}
# LSTM (Custom Implementation)
LSTM_CONFIG = {
    # Note: input_size for nn.LSTM is features per time step (n_series)
    'n_series': len(TARGET_COLS), # Used as input_size for nn.LSTM layer
    'lookback': LOOKBACK_WINDOW, # Used to define sequence length
    'h': PREDICTION_HORIZON, # Prediction horizon
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'output_size': len(TARGET_COLS) # Final output features per step
}
# Autoformer (neuralforecast)
AUTOFORMER_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS),
    'hidden_size': 64,  # d_model
    'n_head': 8,
    'encoder_layers': 2,
    'decoder_layers': 1,
    'd_ff': 256,       # Dimension of feedforward network
    'moving_avg': 25,  # Window size of moving average
    'dropout': 0.05,
    'activation': 'gelu',
    # 'output_attention': False, # Whether to output attention weights
    # 'factor': 3, # AutoCorrelation factor
    # 'scale': None, # Not used currently in neuralforecast's implementation
}
# Informer (neuralforecast)
INFORMER_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS),
    'hidden_size': 64,  # d_model
    'n_head': 8,
    'encoder_layers': 2,
    'decoder_layers': 1,
    'd_ff': 256,
    'dropout': 0.05,
    'activation': 'gelu',
    # 'output_attention': False,
    # 'factor': 5, # ProbSparse Attention factor
    # 'distil': True, # Whether to use distilling in encoder
}
# FEDformer (neuralforecast)
FEDFORMER_CONFIG = {
    'input_size': LOOKBACK_WINDOW,
    'h': PREDICTION_HORIZON,
    'n_series': len(TARGET_COLS),
    'hidden_size': 64,  # d_model
    'n_head': 8,
    'encoder_layers': 2,
    'decoder_layers': 1,
    'd_ff': 256,
    'moving_avg': 25,
    'dropout': 0.05,
    'activation': 'gelu',
    # 'output_attention': False,
    # 'version': 'Fourier', # or 'Wavelets'
    # 'modes': 64, # Number of Fourier modes
    # 'mode_select': 'random', # 'random' or 'low'
}
# --- Update TEACHER/STUDENT Selection ---
# Example: Choose one of the new models
# TEACHER_MODEL_NAME = 'LSTM'
# TEACHER_CONFIG = LSTM_CONFIG.copy()
# STUDENT_MODEL_NAME = 'FEDformer'
# STUDENT_CONFIG = FEDFORMER_CONFIG.copy()
# --- IMPORTANT: Update Dynamic n_series Update ---
# Make sure this part updates ALL relevant model configs
TEACHER_CONFIG['n_series'] = len(TARGET_COLS)
STUDENT_CONFIG['n_series'] = len(TARGET_COLS)
# Add updates for any other configs you defined and might use
NLINEAR_CONFIG['n_series'] = len(TARGET_COLS)
MLP_CONFIG['n_series'] = len(TARGET_COLS)
RNN_CONFIG['n_series'] = len(TARGET_COLS)
RNN_CONFIG['output_size'] = len(TARGET_COLS)
LSTM_CONFIG['n_series'] = len(TARGET_COLS)
LSTM_CONFIG['output_size'] = len(TARGET_COLS)
AUTOFORMER_CONFIG['n_series'] = len(TARGET_COLS)
INFORMER_CONFIG['n_series'] = len(TARGET_COLS)
FEDFORMER_CONFIG['n_series'] = len(TARGET_COLS)
# Adjust input_size/lookback if they differ conceptually
MLP_CONFIG['input_size'] = LOOKBACK_WINDOW
RNN_CONFIG['lookback'] = LOOKBACK_WINDOW
LSTM_CONFIG['lookback'] = LOOKBACK_WINDOW
