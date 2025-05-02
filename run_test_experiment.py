import torch
import os
import sys
import traceback

# --- 导入自定义模块 ---
# 假设脚本在项目根目录运行，可以找到 src
try:
    from src import config as default_config
    from src import utils
    from src.data_handler import load_and_preprocess_data
    from src.models import get_model # 使用通用的 get_model
    from src.trainer import StandardTrainer, RDT_Trainer, get_optimizer, get_loss_function
    from src.schedulers import get_alpha_scheduler, ConstantScheduler
    # from src.evaluator import evaluate_model # 评估可以简化或跳过
except ImportError as e:
    print(f"Error importing modules. Make sure this script is run from the project root directory.")
    print(f"Import Error: {e}")
    sys.exit(1)

def run_test(teacher_model_name, student_model_name, max_epochs=1, device='cpu'):
    """
    运行一个简化的测试实验，只训练指定模型的一个 epoch。
    返回 True 表示成功，False 表示失败。
    """
    print(f"\n--- Running Test: Teacher={teacher_model_name}, Student={student_model_name}, Epochs={max_epochs} ---")
    current_seed = default_config.SEED # 使用默认种子进行测试
    utils.set_seed(current_seed)

    # --- 动态加载配置 ---
    cfg = default_config # 使用默认配置作为基础
    cfg.EPOCHS = max_epochs # 覆盖 epoch
    cfg.DEVICE = device     # 覆盖 device
    cfg.STABILITY_RUNS = 1 # 测试时不进行稳定性运行
    cfg.PATIENCE = 1       # 早停设为1，尽快结束

    # 获取并设置教师模型配置
    teacher_config_name = f"{teacher_model_name.upper()}_CONFIG"
    if hasattr(cfg, teacher_config_name):
        teacher_config = getattr(cfg, teacher_config_name).copy()
        teacher_config['n_series'] = len(cfg.TARGET_COLS)
        if teacher_model_name in ['RNN', 'LSTM']:
            teacher_config['output_size'] = len(cfg.TARGET_COLS)
            teacher_config['lookback'] = cfg.LOOKBACK_WINDOW
            if 'input_size' in teacher_config: del teacher_config['input_size']
        else:
            teacher_config['input_size'] = cfg.LOOKBACK_WINDOW
            if 'lookback' in teacher_config: del teacher_config['lookback']
    else:
        print(f"Error: Config '{teacher_config_name}' not found for teacher model '{teacher_model_name}'.")
        return False

    # 获取并设置学生模型配置
    student_config_name = f"{student_model_name.upper()}_CONFIG"
    if hasattr(cfg, student_config_name):
        student_config = getattr(cfg, student_config_name).copy()
        student_config['n_series'] = len(cfg.TARGET_COLS)
        if student_model_name in ['RNN', 'LSTM']:
            student_config['output_size'] = len(cfg.TARGET_COLS)
            student_config['lookback'] = cfg.LOOKBACK_WINDOW
            if 'input_size' in student_config: del student_config['input_size']
        else:
            student_config['input_size'] = cfg.LOOKBACK_WINDOW
            if 'lookback' in student_config: del student_config['lookback']
    else:
        print(f"Error: Config '{student_config_name}' not found for student model '{student_model_name}'.")
        return False

    try:
        # --- 1. 数据加载 (简化，只获取训练和验证加载器) ---
        print("Loading data...")
        # 注意：如果数据加载本身很慢，测试也会慢
        train_loader, val_loader, _, _ = load_and_preprocess_data(cfg)
        print("Data loaded.")

        # --- 2. 初始化模型 (使用 get_model) ---
        print("Initializing models...")
        teacher_model = get_model(teacher_model_name, teacher_config, is_teacher=True)
        student_model = get_model(student_model_name, student_config, is_teacher=False)
        print("Models initialized.")

        # --- 3. 初始化训练器 (简化，只测试 StandardTrainer) ---
        # 为了通用性，我们只测试 StandardTrainer 是否能运行一个 epoch
        # 如果需要测试 RDT_Trainer，逻辑会更复杂
        print("Initializing trainer...")
        task_loss_fn = get_loss_function(cfg)
        optimizer = get_optimizer(student_model, cfg) # 测试学生模型训练

        # 使用一个临时的保存路径，测试后可以删除
        temp_model_save_path = os.path.join(cfg.RESULTS_DIR, f"temp_test_model_{student_model_name}.pt")

        trainer = StandardTrainer(
            model=student_model,
            train_loader=train_loader,
            val_loader=val_loader, # 需要验证集来完成一个 epoch 的流程
            optimizer=optimizer,
            loss_fn=task_loss_fn,
            device=cfg.DEVICE,
            epochs=cfg.EPOCHS, # 已经是 1 了
            patience=cfg.PATIENCE, # 已经是 1 了
            model_save_path=temp_model_save_path,
            model_name=f"Test_{student_model_name}"
        )
        print("Trainer initialized.")

        # --- 4. 运行一个训练周期 ---
        print(f"Starting training for 1 epoch on {cfg.DEVICE}...")
        _, history = trainer.train()
        print("Training epoch completed.")

        # --- 清理临时文件 ---
        if os.path.exists(temp_model_save_path):
            os.remove(temp_model_save_path)

        # 如果代码运行到这里没有抛出异常，则认为测试通过
        print(f"--- Test PASSED for Teacher={teacher_model_name}, Student={student_model_name} ---")
        return True

    except Exception as e:
        print(f"--- Test FAILED for Teacher={teacher_model_name}, Student={student_model_name} ---")
        print("Error details:")
        traceback.print_exc() # 打印详细的错误堆栈
        return False

if __name__ == "__main__":
    # 允许直接运行此脚本进行单个测试
    import argparse
    parser = argparse.ArgumentParser(description="Run a single test experiment.")
    parser.add_argument('--teacher_model', required=True, help='Teacher model name (e.g., DLinear)')
    parser.add_argument('--student_model', required=True, help='Student model name (e.g., PatchTST)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to run')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to run on')
    args = parser.parse_args()

    # 检查 CUDA 可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU.")
        args.device = 'cpu'

    success = run_test(args.teacher_model, args.student_model, args.epochs, args.device)
    sys.exit(0 if success else 1)