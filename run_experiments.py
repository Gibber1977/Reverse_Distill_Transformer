import os
import subprocess
import pandas as pd
import glob
import re
import time
from datetime import datetime

# --- 配置 ---
DATASETS = {
    "exchange_rate": "./data/exchange_rate.csv",
    "national_illness": "./data/national_illness.csv",
    "weather": "./data/weather.csv",
    "ETTh1": "./data/ETT-small/ETTh1.csv",
    "ETTh2": "./data/ETT-small/ETTh2.csv",
    "ETTm1": "./data/ETT-small/ETTm1.csv",
    "ETTm2": "./data/ETT-small/ETTm2.csv",
}
PRED_LENS = [24, 96, 192, 336, 720]
MODELS = [
    # TaskOnly
    "PatchTST", "DLinear", "NLinear", "LSTM", "Autoformer",
    # Distillation Pairs (Teacher-Student)
    "PatchTST-DLinear", "PatchTST-Autoformer", "PatchTST-PatchTST",
    "DLinear-PatchTST", "DLinear-Autoformer", "DLinear-DLinear"
]
SEQ_LEN = 192
EPOCHS = 100 # 每个实验的 epoch 数
N_RUNS = 5   # 每个实验重复运行次数以计算平均值
RESULTS_DIR = "./results"
SUMMARY_FILENAME_PATTERN = "all_runs_summary_*.csv" # main.py 生成的单个实验汇总文件模式
AGGREGATED_SUMMARY_FILENAME = "all_experiments_summary.csv" # 最终聚合结果文件名

# --- 辅助函数 ---
def parse_model_name(model_str):
    """解析模型字符串，区分教师和学生模型"""
    parts = model_str.split('-')
    if len(parts) == 1:
        # TaskOnly model
        return None, parts[0]
    elif len(parts) == 2:
        # Teacher-Student pair
        return parts[0], parts[1]
    else:
        # 允许模型名称本身包含破折号，但教师-学生分隔符只有一个
        # 例如 "Some-Model-Name" (TaskOnly) vs "Teacher-Model-Student-Model"
        # 暂时按第一个破折号分割，如果需要更复杂的逻辑需要调整
        print(f"警告: 模型名称 '{model_str}' 包含多个破折号，假设第一个为分隔符。")
        return parts[0], '-'.join(parts[1:])
        # 或者抛出错误：
        # raise ValueError(f"无法解析的模型名称格式: {model_str}")

def run_single_experiment(dataset_name, data_path, pred_len, model_str):
    """为单个实验配置运行 main.py"""
    print(f"\n--- 开始运行: 数据集={dataset_name}, 预测步长={pred_len}, 模型={model_str} ---")
    start_time = time.time()

    teacher_model, student_model = parse_model_name(model_str)

    # 构建命令列表 - *** 修改为调用 experiment_launcher.py ***
    cmd = [
        "python", "experiment_launcher.py", # <<< 调用新的启动器
        "--data", dataset_name,
        "--data_path", data_path,
        "--seq_len", str(SEQ_LEN),
        "--pred_len", str(pred_len),
        "--epochs", str(EPOCHS),
        "--n_runs", str(N_RUNS),
        "--output_dir", RESULTS_DIR,
        # 根据模型类型传递参数
    ]

    # --- 根据模型类型传递参数 ---
    cmd.extend(["--model", model_str]) # 传递组合名称用于目录创建等
    cmd.extend(["--student_model", student_model]) # 学生模型总是需要的

    if teacher_model:
        # 只有在 teacher_model 存在时才传递 --teacher_model 参数
        cmd.extend(["--teacher_model", teacher_model])
        # 可以在这里根据需要添加蒸馏类型参数，如果 experiment_launcher 支持的话
        # cmd.extend(["--distillation_type", "rdt"])
    # else:
        # 对于 TaskOnly，不需要传递 --teacher_model

    print(f"执行命令: {' '.join(cmd)}")

    try:
        # 设置PYTHONPATH环境变量，确保能找到src下的模块
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(project_root, 'src')
        if 'PYTHONPATH' in env:
            # 确保 src 路径在最前面，并且处理 Windows 和 Linux 的路径分隔符
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = src_path
        print(f"设置 PYTHONPATH: {env['PYTHONPATH']}") # 调试信息

        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        # process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', env=env) # 不捕获输出，直接打印到控制台

        end_time = time.time()
        print(f"--- 成功完成 (耗时: {end_time - start_time:.2f} 秒): 数据集={dataset_name}, 预测步长={pred_len}, 模型={model_str} ---")
        # print("标准输出:") # 如果 capture_output=True
        # print(process.stdout)
        # print("标准错误:")
        # print(process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"!!! 运行失败 (耗时: {end_time - start_time:.2f} 秒): 数据集={dataset_name}, 预测步长={pred_len}, 模型={model_str} !!!")
        print(f"返回码: {e.returncode}")
        print("标准输出:")
        print(e.stdout) # 打印错误输出帮助调试
        print("标准错误:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"!!! 错误: 找不到 'python' 命令或 'main.py' 文件。请确保 Python 环境已配置且脚本路径正确。")
        return False
    except Exception as e:
        end_time = time.time()
        print(f"!!! 发生未知错误 (耗时: {end_time - start_time:.2f} 秒): 数据集={dataset_name}, 预测步长={pred_len}, 模型={model_str} !!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")
        return False


def aggregate_results():
    """聚合所有实验结果到单个CSV文件"""
    print("\n--- 开始聚合结果 ---")
    all_results_data = []
    # 搜索 results 目录下所有符合模式的 summary 文件
    search_pattern = os.path.join(RESULTS_DIR, "*", "metrics", SUMMARY_FILENAME_PATTERN)
    summary_files = glob.glob(search_pattern, recursive=True)

    if not summary_files:
        print(f"在 '{search_pattern}' 未找到任何结果汇总文件。")
        return

    print(f"找到 {len(summary_files)} 个结果文件进行聚合:")
    for f_path in summary_files:
        print(f"  - {f_path}")
        try:
            # 从文件路径中提取信息
            # 改进正则表达式以处理 Windows 和 Linux 路径分隔符，并更精确匹配
            # 示例: ./results/exchange_rate_DLinear-PatchTST_20250502_221402/metrics/all_runs_summary_20250502_221402.csv
            # 示例: results\weather_PatchTST_20250503_010000\metrics\all_runs_summary_20250503_010000.csv
            match = re.search(r"results[\\/](?P<dataset_model>.*?)_\d{8}_\d{6}[\\/]metrics[\\/]", f_path, re.IGNORECASE)

            if not match:
                print(f"警告: 无法从路径解析信息: {f_path}")
                continue

            dataset_model_str = match.group('dataset_model')

            # 分离数据集和模型名称 (假设最后一个下划线之前是数据集)
            # 这可能需要根据 main.py 的实际命名规则调整
            parts = dataset_model_str.split('_')
            if len(parts) < 2:
                 print(f"警告: 无法从 '{dataset_model_str}' 分离数据集和模型: {f_path}")
                 continue
            model_name_from_path = parts[-1] # 假设模型在最后
            dataset_name_from_path = '_'.join(parts[:-1]) # 其他部分是数据集名称

            # 尝试读取CSV文件
            df_run = pd.read_csv(f_path)

            # 检查必要的列是否存在 (使用 _mean 后缀)
            required_cols = ['pred_len', 'split', 'MSE_mean', 'MAE_mean', 'MAPE_mean', 'WAPE_mean']
            if not all(col in df_run.columns for col in required_cols):
                 print(f"警告: 文件缺少必要列 {required_cols}: {f_path}. 跳过。")
                 print(f"  文件列名: {df_run.columns.tolist()}")
                 continue

            # 只选择测试集的结果进行最终汇总
            df_test = df_run[df_run['split'].str.lower() == 'test'].copy()

            if df_test.empty:
                print(f"警告: 文件中未找到 'test' split 的结果: {f_path}")
                continue

            # 添加数据集和模型信息
            df_test['dataset'] = dataset_name_from_path
            # 使用从路径解析的模型名称，因为它对应文件夹结构
            df_test['model'] = model_name_from_path

            # 选择并重命名列以匹配最终输出格式
            df_agg = df_test[['dataset', 'pred_len', 'model', 'MSE_mean', 'MAE_mean', 'MAPE_mean', 'WAPE_mean']].copy()
            df_agg.rename(columns={
                'pred_len': 'PredictionLength',
                'MSE_mean': 'MSE',
                'MAE_mean': 'MAE',
                'MAPE_mean': 'MAPE',
                'WAPE_mean': 'WAPE'
            }, inplace=True)

            all_results_data.append(df_agg)

        except pd.errors.EmptyDataError:
            print(f"警告: 文件为空: {f_path}")
        except Exception as e:
            print(f"处理文件时出错 {f_path}: {e}")

    if not all_results_data:
        print("没有成功聚合任何结果。")
        return

    # 合并所有数据帧
    final_df = pd.concat(all_results_data, ignore_index=True)

    # 按指定顺序排列
    final_df = final_df[['dataset', 'PredictionLength', 'model', 'MSE', 'MAE', 'MAPE', 'WAPE']]

    # 按数据集、预测长度、模型排序，便于查看
    final_df.sort_values(by=['dataset', 'PredictionLength', 'model'], inplace=True)

    # 保存到最终的汇总文件
    output_path = os.path.join(RESULTS_DIR, AGGREGATED_SUMMARY_FILENAME)
    try:
        final_df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\n--- 结果成功聚合到: {output_path} ---")
    except Exception as e:
        print(f"!!! 保存聚合结果失败: {e} !!!")

# --- 主执行逻辑 ---
def main():
    """主函数，执行所有实验并聚合结果"""
    total_experiments = len(DATASETS) * len(PRED_LENS) * len(MODELS)
    completed_count = 0
    failed_count = 0
    start_time_total = time.time()

    print(f"总共需要运行 {total_experiments} 个实验配置。")

    # 创建结果根目录（如果不存在）
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for dataset_name, data_path in DATASETS.items():
        for pred_len in PRED_LENS:
            for model_str in MODELS:
                completed_count += 1
                print(f"\n>>> 实验 {completed_count}/{total_experiments} <<<")
                success = run_single_experiment(dataset_name, data_path, pred_len, model_str)
                if not success:
                    failed_count += 1

    end_time_total = time.time()
    print(f"\n--- 所有实验运行完成 ---")
    print(f"总耗时: {(end_time_total - start_time_total) / 60:.2f} 分钟")
    print(f"成功运行: {completed_count - failed_count}/{total_experiments}")
    print(f"失败运行: {failed_count}/{total_experiments}")

    # 运行结束后聚合结果
    aggregate_results()

if __name__ == "__main__":
    main()