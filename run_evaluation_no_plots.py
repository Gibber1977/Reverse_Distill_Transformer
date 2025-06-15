import os
import pandas as pd
from datetime import datetime

from src.config import Config
from src.utils import setup_logging, save_results_to_csv, save_experiment_metadata
from src.run_experiment import run_experiment

# --- 快速测试实验配置 ---
DATASETS = {
    'exchange_rate': './data/exchange_rate.csv',
    # 'national_illness': './data/national_illness.csv',
    # 'weather': './data/weather.csv',
    # 'ETTh1': './data/ETT-small/ETTh1.csv',
    # 'ETTh2': './data/ETT-small/ETTh2.csv',
    # 'ETTm1': './data/ETT-small/ETTm1.csv',
    # 'ETTm2': './data/ETT-small/ETTm2.csv',
    # 'PEMS_0':'./data/PEMS_0.csv'
}

# PREDICTION_HORIZONS = [336,720]
PREDICTION_HORIZONS = [336]
LOOKBACK_WINDOW = 192
EPOCHS = 3 # 减少epochs以加快测试
STABILITY_RUNS = 1 # 减少运行次数

# 模型组合: (Teacher, Student)
MODELS = [
    ('DLinear', 'PatchTST'),     # 测试 PatchTST 作为独立学生模型

]

# 噪音注入评估配置 (减少噪音水平)
# NOISE_LEVELS = [0.01,0.02,0.05,0.10,0.15,0.20]
NOISE_LEVELS = []
NOISE_TYPE = 'gaussian'

# 去噪平滑评估配置 (减少平滑系数)
# WEIGHT_SMOOTHING_FACTORS = [0.01,0.02,0.05,0.10,0.15,0.20]
WEIGHT_SMOOTHING_FACTORS = [0.01,0.02]
SMOOTHING_METHOD = 'moving_average'


# --- 主实验函数---

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_quick_test_no_plots/experiments_{timestamp}" # 快速测试结果保存到单独目录
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join("log", f"quick_test_no_plots_experiment_log_{timestamp}.log")
    logger = setup_logging(log_file)

    # 保存实验元数据
    current_config = Config()
    current_config.LOOKBACK_WINDOW = LOOKBACK_WINDOW
    current_config.PREDICTION_HORIZON = PREDICTION_HORIZONS[0]
    current_config.TEACHER_MODEL_NAME = MODELS[0][0]
    current_config.STUDENT_MODEL_NAME = MODELS[0][1]
    current_config.EPOCHS = EPOCHS
    current_config.STABILITY_RUNS = STABILITY_RUNS
    current_config.ROBUSTNESS_NOISE_LEVELS = NOISE_LEVELS
    current_config.WEIGHT_SMOOTHING_FACTORS = WEIGHT_SMOOTHING_FACTORS
    current_config.update_model_configs()
    
    save_experiment_metadata(current_config, results_dir, f"quick_test_no_plots_experiment_{timestamp}")

    all_experiment_results = []
    all_experiment_similarity_results = []

    logger.info("Starting quick evaluation experiments (no plots)...")

    for dataset_name, dataset_path in DATASETS.items():
        for pred_horizon in PREDICTION_HORIZONS:
            for teacher_model, student_model in MODELS:
                logger.info(f"\n--- Running for Dataset: {dataset_name}, Horizon: {pred_horizon}, "
                            f"Models: {teacher_model}/{student_model} ---")

                # 1. 标准评估 (无噪音, 无平滑)
                logger.info("\n--- Running Standard Evaluation ---")
                run_results, sim_results = run_experiment(
                    dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                    teacher_model, student_model, results_dir,
                    experiment_type='standard', logger=logger
                )
                all_experiment_results.extend(run_results)
                all_experiment_similarity_results.extend(sim_results)

                # 2. 噪音注入评估
                logger.info("\n--- Running Noise Injection Evaluation ---")
                for noise_level in NOISE_LEVELS:
                    run_results, sim_results = run_experiment(
                        dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                        teacher_model, student_model, results_dir,
                        noise_level=noise_level, noise_type=NOISE_TYPE,
                        experiment_type='noise_injection', logger=logger
                    )
                    all_experiment_results.extend(run_results)
                    all_experiment_similarity_results.extend(sim_results)

                # 3. 去噪平滑评估
                logger.info("\n--- Running Denoising Smoothing Evaluation ---")
                for weight_smoothing in WEIGHT_SMOOTHING_FACTORS:
                    run_results, sim_results = run_experiment(
                        dataset_name, dataset_path, pred_horizon, LOOKBACK_WINDOW, EPOCHS, STABILITY_RUNS,
                        teacher_model, student_model, results_dir,
                        weight_smoothing=weight_smoothing, smoothing_method=SMOOTHING_METHOD,
                        experiment_type='denoising_smoothing', logger=logger
                    )
                    all_experiment_results.extend(run_results)
                    all_experiment_similarity_results.extend(sim_results)

    # 保存所有结果
    results_df = pd.DataFrame(all_experiment_results)
    similarity_df = pd.DataFrame(all_experiment_similarity_results)

    results_csv_path = os.path.join(results_dir, "quick_test_no_plots_experiment_results.csv")
    similarity_csv_path = os.path.join(results_dir, "quick_test_no_plots_similarity_results.csv")

    save_results_to_csv(results_df, results_csv_path, logger)
    save_results_to_csv(similarity_df, similarity_csv_path, logger)

    logger.info(f"All quick test experiment results saved to: {results_csv_path}")
    logger.info(f"All quick test similarity results saved to: {similarity_csv_path}")

    # Removed visualization generation
    # logger.info("\n--- Generating Quick Test Visualizations ---")
    # plot_noise_evaluation(results_df, similarity_df, results_dir, logger)
    # plot_smoothing_evaluation(results_df, similarity_df, results_dir, logger)
    # logger.info("All quick test visualizations generated.")
    
    logger.info("Quick evaluation experiments (no plots) completed.")

# Removed plot_noise_evaluation function
# Removed plot_smoothing_evaluation function

if __name__ == "__main__":
    main()