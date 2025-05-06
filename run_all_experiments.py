import os
import subprocess
import pandas as pd
import glob
import time
from pathlib import Path

# --- 配置 ---
PYTHON_EXE = 'python' # 或者你的 python 解释器路径
MAIN_SCRIPT = 'main.py'
RESULTS_BASE_DIR = './results'
SUMMARY_FILE = os.path.join(RESULTS_BASE_DIR, 'all_experiments_summary.csv')

# --- 实验参数 ---
DATASETS = {
    'exchange_rate': './data/exchange_rate.csv',
    'national_illness': './data/national_illness.csv',
    'weather': './data/weather.csv',
    'ETTh1': './data/ETT-small/ETTh1.csv',
    'ETTh2': './data/ETT-small/ETTh2.csv',
    'ETTm1': './data/ETT-small/ETTm1.csv',
    'ETTm2': './data/ETT-small/ETTm2.csv',
}

PREDICTION_HORIZONS = [24, 96, 192, 336, 720]
LOOKBACK_WINDOW = 192 # 固定回看窗口
EPOCHS = 100 # 固定 Epochs
STABILITY_RUNS = 5 # 固定运行次数

# 模型组合: (Teacher, Student) 或 (None, Student) for TaskOnly
MODELS = [
    ('Autoformer', 'PatchTST'),
    ('DLinear', 'Autoformer'),
    ('PatchTST', 'DLinear'),
    ('PatchTST', 'Autoformer'),
    ('PatchTST', 'PatchTST'),
    ('DLinear', 'PatchTST'),
    ('DLinear', 'Autoformer'),
    ('DLinear', 'DLinear'),
]

# --- 脚本逻辑 ---
all_results_list = []
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# --- 计算总实验数并初始化计数器 ---
total_experiments = len(DATASETS) * len(PREDICTION_HORIZONS) * len(MODELS)
current_experiment_num = 0
print(f"===== Starting Experiment Run =====")
print(f"Total number of experiment combinations to run: {total_experiments}")
print(f"===================================")


# 获取当前时间戳，用于查找本次运行创建的目录
run_start_time = time.time()

for dataset_name, dataset_path in DATASETS.items():
    for horizon in PREDICTION_HORIZONS:
        for teacher_model, student_model in MODELS:
            current_experiment_num += 1
            print(f"\n===== Running Experiment {current_experiment_num} / {total_experiments} =====")
            print(f"Dataset: {dataset_name}")
            print(f"Horizon: {horizon}")
            print(f"Teacher: {teacher_model if teacher_model else 'None'}")
            print(f"Student: {student_model}")
            print(f"Lookback: {LOOKBACK_WINDOW}")
            print(f"Epochs: {EPOCHS}")
            print(f"Runs: {STABILITY_RUNS}")
            print(f"-----------------------------")

            # 构建命令行参数
            cmd = [
                PYTHON_EXE,
                MAIN_SCRIPT,
                '--dataset_path', dataset_path,
                '--prediction_horizon', str(horizon),
                '--lookback_window', str(LOOKBACK_WINDOW),
                '--epochs', str(EPOCHS),
                '--stability_runs', str(STABILITY_RUNS),
                '--student_model_name', student_model,
            ]
            if teacher_model:
                cmd.extend(['--teacher_model_name', teacher_model])
            else:
                # 对于 TaskOnly，明确传递 None
                cmd.extend(['--teacher_model_name', 'None'])

            # 执行 main.py
            try:
                # Run main.py and let its output stream to the console
                process = subprocess.run(cmd, check=True, encoding='utf-8') # Removed capture_output=True, text=True
                # Output is now printed directly by main.py
                # Check for errors specifically if the process fails
                if process.returncode != 0 and process.stderr:
                     print("--- main.py Error Output (captured on failure) ---")
                     print(process.stderr)
                     print("----------------------------------------------------")

                # --- 查找对应的结果目录和 average_metrics 文件 ---
                # 构造预期的目录名前缀
                teacher_part = teacher_model if teacher_model else "NoTeacher"
                # 处理 ETT 数据集名称
                if 'ETT-small' in dataset_path:
                    parent_dir_name = Path(dataset_path).parent.name
                    dataset_prefix = f"{parent_dir_name}_{Path(dataset_path).stem}"
                else:
                    dataset_prefix = Path(dataset_path).stem

                expected_dir_prefix = f"{dataset_prefix}_{teacher_part}_{student_model}_h{horizon}_"
                print(f"Searching for results directory starting with: {expected_dir_prefix}")

                # 查找在本次运行开始后创建的最新匹配目录
                matching_dirs = []
                for item in Path(RESULTS_BASE_DIR).iterdir():
                    if item.is_dir() and item.name.startswith(expected_dir_prefix):
                         try:
                             # 使用修改时间 (st_mtime) 可能更可靠，因为它反映了目录内容的最后更改
                             dir_mod_time = item.stat().st_mtime
                             if dir_mod_time >= run_start_time:
                                 matching_dirs.append((item, dir_mod_time))
                         except OSError as oe:
                             print(f"Warning: Could not stat directory {item}: {oe}")
                             continue # Ignore potential permission errors

                if not matching_dirs:
                    print(f"Error: Could not find results directory for {expected_dir_prefix} created/modified after script start.")
                    continue

                # 按修改时间排序，获取最新的目录
                matching_dirs.sort(key=lambda x: x[1], reverse=True)
                latest_experiment_dir = matching_dirs[0][0]
                metrics_dir = latest_experiment_dir / 'metrics'
                print(f"Found results directory: {latest_experiment_dir}")

                # 查找 average_metrics 文件
                avg_metrics_files = list(metrics_dir.glob('average_metrics_*.csv'))

                if not avg_metrics_files:
                    print(f"Error: average_metrics_*.csv not found in {metrics_dir}")
                    continue

                avg_metrics_file_path = avg_metrics_files[0] # 假设只有一个
                print(f"Found average metrics file: {avg_metrics_file_path}")

                # 读取 metrics 文件
                df_metrics = pd.read_csv(avg_metrics_file_path)

                # 添加实验参数列
                df_metrics['dataset'] = dataset_name
                df_metrics['horizon'] = horizon
                # 创建统一的模型名称列
                model_combo_name = f"{teacher_model}-{student_model}" if teacher_model else student_model
                df_metrics['model_combination'] = model_combo_name

                # 重新排序列，使参数列在前
                cols = ['dataset', 'horizon', 'model_combination', 'split', 'model_type', 'metric', 'value']
                # 确保所有列都存在，以防万一
                df_metrics = df_metrics.reindex(columns=cols)


                all_results_list.append(df_metrics)
                print(f"Successfully processed results for this combination.")

            except subprocess.CalledProcessError as e:
                print(f"Error running main.py for {dataset_name} h={horizon} {teacher_model}-{student_model}")
                print(f"Return code: {e.returncode}")
                print("--- Error Output ---")
                # 打印完整的 stderr 以便调试
                print(e.stderr)
                print("--------------------")
            except FileNotFoundError:
                 print(f"Error: '{PYTHON_EXE}' command not found. Please ensure Python is in your PATH.")
                 break # Stop the script if Python cannot be found
            except Exception as e:
                print(f"An unexpected error occurred during processing {dataset_name} h={horizon} {teacher_model}-{student_model}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for unexpected errors


# --- 合并并保存所有结果 ---
if all_results_list:
    final_summary_df = pd.concat(all_results_list, ignore_index=True)
    try:
        final_summary_df.to_csv(SUMMARY_FILE, index=False, float_format='%.6f')
        print(f"\n===== All Experiments Finished =====")
        print(f"Summary results saved to: {SUMMARY_FILE}")
    except Exception as e:
        print(f"\nError saving final summary CSV: {e}")
else:
    print("\n===== No results collected. Summary file not created. =====")