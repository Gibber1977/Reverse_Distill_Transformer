import torch
import os

# --- 基本配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 获取项目根目录 rdt_framework/
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# --- 数据集配置 ---
DATASET_PATH = os.path.join(DATA_DIR, 'weatherHistory.csv')
DATE_COL = 'Formatted Date'         # 日期/时间列名
TARGET_COLS = ['Temperature (C)']   # 需要预测的目标列（可以是多个）
# TARGET_COLS = ['Temperature (C)', 'Humidity'] # 示例：多变量预测
TIME_FREQ = 'H'                     # 时间序列频率 ('H' for hourly) - 需要根据你的数据调整

# --- 数据处理配置 ---
LOOKBACK_WINDOW = 96     # 回看窗口大小 (e.g., 4 days for hourly data)
PREDICTION_HORIZON = 96   # 预测未来步长 (e.g., 1 day for hourly data)
VAL_SPLIT_RATIO = 0.2     # 验证集比例 (从训练集中划分)
TEST_SPLIT_RATIO = 0.2    # 测试集比例 (从总数据末尾划分)
BATCH_SIZE = 128         # 批次大小 (根据 GPU 内存调整)
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
EPOCHS = 100              # 最大训练轮数
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
ROBUSTNESS_NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.2] # 添加到测试集的高斯噪声标准差比例

STABILITY_RUNS = 10      # 运行多次以评估稳定性 (设为 1 则不进行稳定性评估)

# --- 实验管理 ---
SEED = 42               # 随机种子，用于可复现性
EXPERIMENT_NAME = f"RDT_{STUDENT_MODEL_NAME}_vs_{TEACHER_MODEL_NAME}_h{PREDICTION_HORIZON}"

# --- 创建结果目录 ---
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 动态更新模型配置中的 n_series ---
TEACHER_CONFIG['n_series'] = len(TARGET_COLS)
STUDENT_CONFIG['n_series'] = len(TARGET_COLS)
