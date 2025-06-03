import torch
import os
from datetime import datetime

class Config:
    def __init__(self):
        # --- 基本配置 ---
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'results')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'log')
        self.PLOTS_DIR = os.path.join(self.RESULTS_DIR, 'plots')

        # --- 数据集配置 ---
        self.DATASET_PATH = os.path.join(self.DATA_DIR, 'national_illness.csv')
        self.DATE_COL = 'date'
        self.TARGET_COLS = ['OT']
        self.EXOGENOUS_COLS = [] # 例如: ['0', '1', '2', '3', '4', '5', '6'] 或留空表示不使用额外协变量
        self.TIME_FREQ = 'h' # 默认值，将在实验脚本中动态设置
        self.TIME_ENCODING_TYPE = 'linear' # 'linear' or 'cyclic'

        self.DATASET_TIME_FREQ_MAP = {
            'ETTh1.csv': 'h',
            'ETTh2.csv': 'h',
            'ETTm1.csv': 'min',
            'ETTm2.csv': 'min',
            'exchange_rate.csv': 'd',
            'national_illness.csv': 'w',
            'weather.csv': 'min',
            # ETT-small 数据集路径可能需要调整以匹配实际文件名
            'data/ETT-small/ETTh1.csv': 'h',
            'data/ETT-small/ETTh2.csv': 'h',
            'data/ETT-small/ETTm1.csv': 'min',
            'data/ETT-small/ETTm2.csv': 'min',
            'PEMS_0.csv':'min',
        }

        # --- 数据处理配置 ---
        self.LOOKBACK_WINDOW = 336
        self.PREDICTION_HORIZON = 192
        self.VAL_SPLIT_RATIO = 0.2
        self.TEST_SPLIT_RATIO = 0.2
        self.BATCH_SIZE = 32 # Increased for better GPU utilization
        self.NUM_WORKERS = 0  # Increased for faster data loading

        # --- 模型配置 ---
        self.TEACHER_MODEL_NAME = 'DLinear'
        self.TEACHER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 64,
            'moving_avg_window': 25,
        }

        self.STUDENT_MODEL_NAME = 'PatchTST'
        self.STUDENT_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'patch_len': 16,
            'stride': 8,
            'n_layers': 3,
            'n_heads': 4,
            'hidden_size': 32,
            'ff_hidden_size': 64,
            'revin': True,
            'dropout': 0.3,       # Default dropout for PatchTST
            'head_dropout': 0.0,  # Default head_dropout for PatchTST
        }

        # --- 训练配置 ---
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.EPOCHS = 2
        self.LEARNING_RATE = 0.001
        self.OPTIMIZER = 'Adam'
        self.WEIGHT_DECAY = 1e-5
        self.PATIENCE = 50
        self.LOSS_FN = 'MSE'
        self.USE_AMP = True # Enable Automatic Mixed Precision (AMP)

        # --- RDT 配置 ---
        self.ALPHA_START = 0.3
        self.ALPHA_END = 0.7
        self.ALPHA_SCHEDULE = 'linear'
        self.CONSTANT_ALPHA = 0.5

        # --- 评估配置 ---
        self.METRICS = ['mae', 'mse']
        self.PLOT_EVALUATION_DETAILS = False # 是否绘制详细的评估图表
        self.ROBUSTNESS_NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]

        # --- 新增数据处理和评估配置 ---
        self.TRAIN_NOISE_INJECTION_LEVEL = 0.0
        self.VAL_NOISE_INJECTION_LEVEL = 0.0
        self.NOISE_TYPE = 'gaussian'

        self.SMOOTHING_METHOD = 'none'
        self.SMOOTHING_FACTOR = 0.5
        self.SMOOTHING_APPLY_TRAIN = False
        self.SMOOTHING_APPLY_VAL = False
        self.SMOOTHING_APPLY_TEST = False
        self.SMOOTHING_WEIGHT_SMOOTHING = 0.0

        self.SIMILARITY_METRIC = 'cosine_similarity'
        self.N_FEATURES = None

        self.STABILITY_RUNS = 1

        # --- 实验管理 ---
        self.SEED = 42
        self.EXPERIMENT_NAME = f"RDT_{self.STUDENT_MODEL_NAME}_vs_{self.TEACHER_MODEL_NAME}_h{self.PREDICTION_HORIZON}"

        # --- 日志配置 ---
        self.LOG_LEVEL = 'INFO'
        self.LOG_FILE_NAME = f"{self.EXPERIMENT_NAME}_{os.path.basename(self.DATASET_PATH).split('.')[0]}.log"
        self.LOG_FILE_PATH = os.path.join(self.LOG_DIR, self.LOG_FILE_NAME)

        # --- 新增模型参数设置 ---
        self.NLINEAR_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        }
        self.MLP_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 512,
            'num_layers': 2,
            'activation': 'relu',
            'dropout': 0.1
        }
        self.RNN_CONFIG = {
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1,
        }
        self.LSTM_CONFIG = {
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1,
        }
        self.AUTOFORMER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 64,
            'n_head': 8,
            'encoder_layers': 2,
            'decoder_layers': 1,
            'd_ff': 256,
            'moving_avg': 25,
            'dropout': 0.05,
            'activation': 'gelu',
        }
        self.INFORMER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 64,
            'n_head': 8,
            'encoder_layers': 2,
            'decoder_layers': 1,
            'd_ff': 256,
            'dropout': 0.05,
            'activation': 'gelu',
        }
        self.FEDFORMER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 64,
            'n_head': 8,
            'encoder_layers': 2,
            'decoder_layers': 1,
            'd_ff': 256,
            'moving_avg': 25,
            'dropout': 0.05,
            'activation': 'gelu',
        }

        # Dynamic updates (now within the class)
        self.MLP_CONFIG['input_size'] = self.LOOKBACK_WINDOW
        self.RNN_CONFIG['lookback'] = self.LOOKBACK_WINDOW
        self.LSTM_CONFIG['lookback'] = self.LOOKBACK_WINDOW

    def update_model_configs(self):
        """
        在 LOOKBACK_WINDOW 或 PREDICTION_HORIZON 改变后，
        动态更新所有依赖这些值的模型配置。
        """
        # 更新教师模型配置
        self.TEACHER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        # 更新学生模型配置
        self.STUDENT_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        # 更新其他模型配置
        self.NLINEAR_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        self.MLP_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        self.RNN_CONFIG.update({
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        self.LSTM_CONFIG.update({
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        self.AUTOFORMER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        self.INFORMER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        self.FEDFORMER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
        })
        # 确保 EXPERIMENT_NAME 也更新
        self.EXPERIMENT_NAME = f"RDT_{self.STUDENT_MODEL_NAME}_vs_{self.TEACHER_MODEL_NAME}_h{self.PREDICTION_HORIZON}"
        self.LOG_FILE_NAME = f"{self.EXPERIMENT_NAME}_{os.path.basename(self.DATASET_PATH).split('.')[0]}.log"
        self.LOG_FILE_PATH = os.path.join(self.LOG_DIR, self.LOG_FILE_NAME)
