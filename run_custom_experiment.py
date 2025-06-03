import os
import json
import torch
from src.config import Config
from src.data_handler import load_and_preprocess_data
from src.models import get_model
from src.trainer import StandardTrainer
from src.evaluator import evaluate_model
from src.utils import setup_logging, set_seed, plot_training_metrics, plot_weights_biases_distribution
from src.trainer import get_optimizer, get_loss_function

def main():
    # 1. 初始化Config实例
    config = Config()

    # 2. 更新配置参数
    config.DATASET_PATH = os.path.join(config.DATA_DIR, 'PEMS_0.csv')
    config.LOOKBACK_WINDOW = 336
    config.VAL_SPLIT_RATIO = 0.2
    config.TEST_SPLIT_RATIO = 0.2
    config.METRICS = ['mae', 'mse']
    config.RESULTS_DIR = os.path.join(config.BASE_DIR, 'results0603', 'Part1')
    config.TEACHER_MODEL_NAME = None  # 明确设置为None，表示没有Teacher_Model
    config.STUDENT_MODEL_NAME = None # 确保学生模型名称在循环中设置
    config.update_model_configs() # 调用config.update_model_configs()来更新所有依赖LOOKBACK_WINDOW和PREDICTION_HORIZON的模型配置。

    # 3. 设置日志
    logger = setup_logging(config.LOG_FILE_PATH, config.LOG_LEVEL)

    # 4. 加载和预处理数据
    logger.info(f"Loading and preprocessing data from {config.DATASET_PATH}")
    
    # 获取数据集文件名以确定时间频率
    dataset_filename = os.path.basename(config.DATASET_PATH)
    time_freq = config.DATASET_TIME_FREQ_MAP.get(dataset_filename, config.TIME_FREQ)
    config.TIME_FREQ = time_freq # 更新config中的TIME_FREQ

    # 5. 定义模型列表和预测窗口
    models_to_run = ['DLinear', 'LSTM', 'NLinear', 'PatchTST']
    prediction_horizons = [336, 720]

    # 6. 实验循环
    for h in prediction_horizons:
        config.PREDICTION_HORIZON = h
        logger.info(f"--- Iteration for PREDICTION_HORIZON = {h} ---")

        logger.info(f"Loading and preprocessing data with PREDICTION_HORIZON = {h}")
        train_loader, val_loader, test_loader, scaler, n_features_current_h = load_and_preprocess_data(
            config.DATASET_PATH,
            config, # config object now contains config.PREDICTION_HORIZON = h
            logger,
            time_freq
        )
        config.N_FEATURES = n_features_current_h
        
        config.update_model_configs() # IMPORTANT: Update model configs based on new h and N_FEATURES

        for model_name in models_to_run:
            config.STUDENT_MODEL_NAME = model_name
            config.update_model_configs() # IMPORTANT: Update for specific model name

            logger.info(f"--- Running experiment for Model: {model_name}, Horizon: {h} ---")
            set_seed(config.SEED)

            # 检查并设置设备
            if torch.cuda.is_available() and config.DEVICE == 'cuda':
                config.DEVICE = 'cuda'
                logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                config.DEVICE = 'cpu'
                logger.info("CUDA is not available. Using CPU.")

            # 获取模型
            model = get_model(model_name, config)
            model.to(config.DEVICE) # 将模型移动到指定设备
            logger.info(f"Model {model_name} initialized and moved to {config.DEVICE}.")

            # 获取优化器和损失函数
            optimizer = get_optimizer(model, config)
            loss_fn = get_loss_function(config)

            # 定义模型保存路径
            model_save_dir = os.path.join(config.RESULTS_DIR, f"{model_name}_h{h}")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, "best_model.pt")

            # 训练模型
            trainer = StandardTrainer(model, train_loader, val_loader, optimizer, loss_fn, config.DEVICE, config.EPOCHS, config.PATIENCE, model_save_path, scaler, config, model_name=model_name)
            trained_model, history = trainer.train() # trainer.train() 现在返回模型和历史记录
            logger.info(f"Training complete for {model_name} with horizon {h}. Best model saved at {model_save_path}")

            # 绘制训练指标曲线
            plot_save_dir = os.path.join(config.PLOTS_DIR, f"{model_name.lower()}_h{h}")
            os.makedirs(plot_save_dir, exist_ok=True)
            plot_training_metrics(history, plot_save_dir, model_name)
            plot_weights_biases_distribution(trained_model, plot_save_dir, model_name)

            # 评估模型
            # trained_model 已经是加载好的最佳模型，无需再次加载
            test_metrics, true_values, predictions = evaluate_model(trained_model, test_loader, config.DEVICE, scaler, config, logger, model_name=model_name)
            logger.info(f"Test metrics for {model_name} (h={h}): {test_metrics}")

            # 保存结果
            results_dir = os.path.join(config.RESULTS_DIR, f"{model_name}_h{h}")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, "run_metrics.json")
            with open(results_file, 'w') as f:
                json.dump(test_metrics, f, indent=4)
            logger.info(f"Results saved to {results_file}")

# 主执行块
if __name__ == "__main__":
    main()