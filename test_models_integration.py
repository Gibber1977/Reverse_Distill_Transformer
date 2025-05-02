import os
import sys
import re
import importlib
import torch # Import torch to check for CUDA

# 动态导入 run_test 函数
try:
    # 确保 Python 路径包含当前目录，以便找到 run_test_experiment
    sys.path.insert(0, os.path.dirname(__file__))
    run_test_module = importlib.import_module("run_test_experiment")
    run_test = run_test_module.run_test
except ImportError as e:
    print(f"Error: Could not import 'run_test_experiment.py'. Make sure it exists in the project root.")
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# 获取 config.py 中定义的模型配置字典名称 (去除 'CONFIG')
project_root = os.path.dirname(__file__)
config_path = os.path.join(project_root, 'src', 'config.py')
model_configs = []
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 使用正则表达式查找所有以 _CONFIG = { 结尾的变量名
        matches = re.findall(r'^([A-Z0-9_]+)_CONFIG\s*=\s*\{', content, re.MULTILINE)
        for model_name in matches:
            # 排除 TEACHER 和 STUDENT 本身，只取具体模型配置
            if model_name not in ['TEACHER', 'STUDENT']:
                 model_configs.append(model_name) # 保留大写名称

except FileNotFoundError:
    print(f"Error: Cannot find config file at {config_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading config file: {e}")
    sys.exit(1)

# 确保提取到了模型名称
if not model_configs:
    print("Error: Could not extract any model config names (e.g., MLP_CONFIG) from src/config.py")
    sys.exit(1)

# 假设 run_test 函数接受 get_model 期望的大小写名称
# 创建一个映射，将大写配置名称转换为 get_model 期望的名称
model_name_map = {
    "NLINEAR": "NLinear",
    "MLP": "MLP",
    "RNN": "RNN",
    "LSTM": "LSTM",
    "AUTOFORMER": "Autoformer",
    "INFORMER": "Informer",
    "FEDFORMER": "FEDformer",
    # 添加其他模型（如果需要）
    "DLINEAR": "DLinear", # 确保默认模型也能被测试（如果需要）
    "PATCHTST": "PatchTST" # 确保默认模型也能被测试（如果需要）
}
models_to_test = [model_name_map.get(cfg_name) for cfg_name in model_configs if model_name_map.get(cfg_name)]
print(f"Found models to test (mapped to expected names): {models_to_test}")

print("\nStarting model integration tests (running 1 epoch for each model using run_test_experiment.py)...")
results = {}
failed_models = []

# 确定测试设备 (优先使用 CUDA)
test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {test_device}")

for model_name_mapped in models_to_test:
    print(f"\n--- Testing Model: {model_name_mapped} (as both Teacher and Student) ---")
    try:
        # 调用 run_test_experiment 中的函数
        # 将教师和学生都设置为当前模型进行简单测试
        success = run_test(
            teacher_model_name=model_name_mapped, # 使用映射后的名称
            student_model_name=model_name_mapped, # 使用映射后的名称
            max_epochs=1,
            device=test_device
        )

        if success:
            print(f"Model '{model_name}' integration test PASSED.")
            results[model_name] = 'PASSED'
        else:
            print(f"Model '{model_name}' integration test FAILED. Check output above.")
            results[model_name] = 'FAILED'
            failed_models.append(model_name)

    except Exception as e:
        print(f"An unexpected error occurred while running the test function for {model_name}: {e}")
        results[model_name] = f'FAILED (Exception in test script: {e})'
        failed_models.append(model_name)

print("\n--- Test Summary ---")
for model, status in results.items():
    print(f"{model}: {status}")

if not failed_models:
    print("\nAll model integration tests passed!")
    sys.exit(0)
else:
    print(f"\nIntegration tests failed for the following models: {', '.join(failed_models)}")
    sys.exit(1) # 以非零状态退出，表示测试失败