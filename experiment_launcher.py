import argparse
import importlib
import os
import sys
import torch
import numpy as np # For np.nan if needed

# --- 确保 src 目录在 Python 路径中 ---
# 这使得我们可以导入 main 和 src.config
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# --- ----------------------------- ---

# 尝试导入重构后的 main 模块中的函数
try:
    import main as experiment_runner
except ImportError as e:
    print(f"Error: Could not import 'main' module. Make sure main.py is in the project root.")
    print(f"Details: {e}")
    sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Experiment Launcher for RDT Framework")

    # --- 从 run_experiments.py 传入的参数 ---
    parser.add_argument('--data', type=str, required=True, help='Dataset name (e.g., exchange_rate)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--seq_len', type=int, required=True, help='Lookback window size')
    parser.add_argument('--pred_len', type=int, required=True, help='Prediction horizon')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--n_runs', type=int, required=True, help='Number of stability runs')
    parser.add_argument('--output_dir', type=str, required=True, help='Base directory for saving results')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., PatchTST or Teacher-Student)')
    parser.add_argument('--teacher_model', type=str, help='Teacher model name (if using distillation)')
    parser.add_argument('--student_model', type=str, required=True, help='Student model name (or the only model if not distilling)')

    # --- 可选的覆盖参数 ---
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Override device (default from config)')
    parser.add_argument('--seed', type=int, help='Override base random seed (default from config)')
    parser.add_argument('--batch_size', type=int, help='Override batch size (default from config)')
    parser.add_argument('--lr', type=float, help='Override learning rate (default from config)')
    parser.add_argument('--patience', type=int, help='Override early stopping patience (default from config)')
    parser.add_argument('--loss_fn', type=str, choices=['MSE', 'MAE'], help='Override loss function (default from config)')
    parser.add_argument('--target_cols', type=str, help='Override target columns (comma-separated, e.g., "OT,Value")')


    return parser.parse_args()

def load_and_update_config(args):
    """加载默认配置并根据命令行参数更新"""
    # 动态加载默认配置模块
    try:
        config_module = importlib.import_module('src.config')
        # 使用 vars() 获取模块的字典表示，然后创建 Namespace
        # 需要过滤掉非配置项，或者确保 config.py 只包含配置变量
        default_config_dict = {k: v for k, v in vars(config_module).items() if not k.startswith('__') and not callable(v) and not isinstance(v, type(sys))}
        cfg = argparse.Namespace(**default_config_dict)
    except ImportError:
        print("Error: Cannot find src/config.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config from src.config: {e}")
        sys.exit(1)


    # --- 使用 args 更新 cfg ---
    cfg.DATASET_PATH = args.data_path
    cfg.LOOKBACK_WINDOW = args.seq_len
    cfg.PREDICTION_HORIZON = args.pred_len
    cfg.EPOCHS = args.epochs
    cfg.STABILITY_RUNS = args.n_runs
    cfg.RESULTS_DIR = args.output_dir # Base directory
    cfg.TEACHER_MODEL_NAME = args.teacher_model # 可能为 None
    cfg.STUDENT_MODEL_NAME = args.student_model

    # 添加从 args 解析的名称，用于目录创建和日志记录
    cfg.dataset_name = args.data
    cfg.model_name = args.model # 完整名称，如 "PatchTST" 或 "DLinear-PatchTST"

    # 可选覆盖
    if args.device: cfg.DEVICE = args.device
    if args.seed: cfg.SEED = args.seed
    if args.batch_size: cfg.BATCH_SIZE = args.batch_size
    if args.lr: cfg.LEARNING_RATE = args.lr
    if args.patience: cfg.PATIENCE = args.patience
    if args.loss_fn: cfg.LOSS_FN = args.loss_fn
    if args.target_cols:
        cfg.TARGET_COLS = [col.strip() for col in args.target_cols.split(',')]
        # --- 重要: 更新依赖 TARGET_COLS 的 n_series ---
        n_series = len(cfg.TARGET_COLS)
        print(f"Updating n_series based on target_cols: {n_series}")
        model_configs_to_update = [
            'TEACHER_CONFIG', 'STUDENT_CONFIG', 'NLINEAR_CONFIG', 'MLP_CONFIG',
            'RNN_CONFIG', 'LSTM_CONFIG', 'AUTOFORMER_CONFIG', 'INFORMER_CONFIG',
            'FEDFORMER_CONFIG'
        ]
        for config_name in model_configs_to_update:
             if hasattr(cfg, config_name):
                 model_cfg_dict = getattr(cfg, config_name)
                 if isinstance(model_cfg_dict, dict):
                     model_cfg_dict['n_series'] = n_series
                     # 特殊处理 RNN/LSTM 的 output_size
                     if config_name in ['RNN_CONFIG', 'LSTM_CONFIG']:
                         model_cfg_dict['output_size'] = n_series


    # --- 动态更新依赖于 LOOKBACK/HORIZON 的模型配置 ---
    model_configs_to_update = [
        'TEACHER_CONFIG', 'STUDENT_CONFIG', 'NLINEAR_CONFIG', 'MLP_CONFIG',
        'RNN_CONFIG', 'LSTM_CONFIG', 'AUTOFORMER_CONFIG', 'INFORMER_CONFIG',
        'FEDFORMER_CONFIG'
    ]
    for config_name in model_configs_to_update:
        if hasattr(cfg, config_name):
            model_cfg = getattr(cfg, config_name)
            if isinstance(model_cfg, dict): # 确保它是字典
                # DLinear/NLinear/PatchTST/Autoformer/Informer/FEDformer 使用 input_size
                if 'input_size' in model_cfg: model_cfg['input_size'] = cfg.LOOKBACK_WINDOW
                # MLP/RNN/LSTM 可能使用不同的命名或概念
                if config_name == 'MLP_CONFIG' and 'input_size' in model_cfg: model_cfg['input_size'] = cfg.LOOKBACK_WINDOW
                if config_name in ['RNN_CONFIG', 'LSTM_CONFIG'] and 'lookback' in model_cfg: model_cfg['lookback'] = cfg.LOOKBACK_WINDOW

                if 'h' in model_cfg: model_cfg['h'] = cfg.PREDICTION_HORIZON


    # 确保 DEVICE 设置有效
    if not hasattr(cfg, 'DEVICE'): cfg.DEVICE = 'cuda' # Provide default if missing
    if cfg.DEVICE == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested via --device or config, but not available. Falling back to CPU.")
        cfg.DEVICE = 'cpu'

    # 确保 METRICS 存在且包含所需指标 (假设 evaluator 会计算它们)
    if not hasattr(cfg, 'METRICS') or not cfg.METRICS:
        cfg.METRICS = ['mse', 'mae', 'mape', 'wape'] # Default required metrics
    else:
        # Ensure required metrics are present
        required = ['mse', 'mae', 'mape', 'wape']
        for m in required:
            if m not in cfg.METRICS:
                cfg.METRICS.append(m)


    return cfg


if __name__ == "__main__":
    args = parse_arguments()
    config = load_and_update_config(args)

    # 调用 main.py 中重构的函数
    try:
        experiment_runner.run_experiment_suite(config)
    except AttributeError:
         print("Error: Function 'run_experiment_suite' not found in 'main' module.")
         print("Please ensure you have saved the refactored main.py.")
         sys.exit(1)
    except Exception as e:
        print(f"An error occurred during experiment execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nExperiment Launcher Finished.")