import os
import subprocess
import time
from pathlib import Path

# --- 配置 ---
PYTHON_EXE = 'python' # 或者你的 python 解释器路径
MAIN_SCRIPT = 'main.py'
RESULTS_BASE_DIR = './results_quick_test' # 使用单独的结果目录

# --- 测试参数 ---
# 使用导致错误的特定配置或一个快速运行的配置
DATASET_NAME = 'weather' # 从错误日志中看是 weather 数据集
DATASET_PATH = './data/weather.csv' # 假设 weather.csv 存在于 data 目录
PREDICTION_HORIZON = 24
LOOKBACK_WINDOW = 96 # 使用一个合理的回看窗口，可以根据需要调整
EPOCHS = 1 # 仅运行 1 个 epoch 以快速检查
STABILITY_RUNS = 1 # 仅运行 1 次
TEACHER_MODEL = None # TaskOnly 模式
STUDENT_MODEL = 'Autoformer' # 导致错误的模型

# --- 脚本逻辑 ---
os.makedirs(RESULTS_BASE_DIR, exist_ok=True) # 创建测试结果目录

print(f"===== Starting Quick Test Run =====")
print(f"Dataset: {DATASET_NAME}")
print(f"Horizon: {PREDICTION_HORIZON}")
print(f"Teacher: {TEACHER_MODEL if TEACHER_MODEL else 'None'}")
print(f"Student: {STUDENT_MODEL}")
print(f"Lookback: {LOOKBACK_WINDOW}")
print(f"Epochs: {EPOCHS}")
print(f"Runs: {STABILITY_RUNS}")
print(f"-----------------------------")

# 构建命令行参数
cmd = [
    PYTHON_EXE,
    MAIN_SCRIPT,
    '--dataset_path', DATASET_PATH,
    '--prediction_horizon', str(PREDICTION_HORIZON),
    '--lookback_window', str(LOOKBACK_WINDOW),
    '--epochs', str(EPOCHS),
    '--stability_runs', str(STABILITY_RUNS),
    '--student_model_name', STUDENT_MODEL
    # 移除 --results_dir 参数，main.py 会自动生成目录
]
if TEACHER_MODEL:
    cmd.extend(['--teacher_model_name', TEACHER_MODEL])
else:
    # 对于 TaskOnly，明确传递 None
    cmd.extend(['--teacher_model_name', 'None'])

# 执行 main.py
try:
    print(f"Executing command: {' '.join(cmd)}")
    # 运行 main.py 并让其输出流到控制台
    process = subprocess.run(cmd, check=True, encoding='utf-8')
    print("\n===== Quick Test Completed Successfully =====")
    print("The script ran without raising the previous KeyError.")

except subprocess.CalledProcessError as e:
    print(f"\n===== Quick Test FAILED =====")
    print(f"Error running main.py for the test configuration.")
    print(f"Return code: {e.returncode}")
    print("--- Error Output ---")
    # 打印完整的 stderr 以便调试
    if e.stderr:
        print(e.stderr)
    else:
        print("(No stderr captured, check console output above for errors)")
    print("--------------------")
    print("The KeyError might still exist or another error occurred.")

except FileNotFoundError:
     print(f"Error: '{PYTHON_EXE}' command not found. Please ensure Python is in your PATH.")
except Exception as e:
    print(f"\n===== Quick Test FAILED (Unexpected Error) =====")
    print(f"An unexpected error occurred during the test run: {e}")
    import traceback
    traceback.print_exc()