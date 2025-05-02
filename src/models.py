import torch.nn as nn
from neuralforecast.models import DLinear, PatchTST
from src import config

def get_model(model_name, model_config, is_teacher=False):
    """根据名称和配置获取模型实例"""
    print(f"Initializing {'Teacher' if is_teacher else 'Student'} model: {model_name}")
    print(f"Model Config: {model_config}")

    # 确保配置中的关键参数与全局配置一致
    model_config['input_size'] = config.LOOKBACK_WINDOW
    model_config['h'] = config.PREDICTION_HORIZON
    model_config['n_series'] = len(config.TARGET_COLS)

    if model_name == 'DLinear':
        # DLinear 需要 'input_size', 'h', 'n_series' 等参数
        model = DLinear(**model_config)
    elif model_name == 'PatchTST':
        # PatchTST 需要 'input_size', 'h', 'n_series', 'patch_len', 'stride' 等
        model = PatchTST(**model_config)
    # --- 可以添加其他模型的选择 ---
    # elif model_name == 'NLinear':
    #     model = NLinear(**model_config)
    # elif model_name == 'LSTM': # 需要自定义实现或包装
    #     # model = YourLSTMImplementation(...)
    #     raise NotImplementedError("LSTM model not implemented yet")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} model initialized with {num_params:,} trainable parameters.")

    return model

def get_teacher_model(cfg):
    """获取教师模型实例"""
    return get_model(cfg.TEACHER_MODEL_NAME, cfg.TEACHER_CONFIG.copy(), is_teacher=True)

def get_student_model(cfg):
    """获取学生模型实例"""
    return get_model(cfg.STUDENT_MODEL_NAME, cfg.STUDENT_CONFIG.copy(), is_teacher=False)

# --- 可以在这里添加自定义模型实现 ---
# class YourCustomModel(nn.Module):
#     def __init__(self, ...):
#         super().__init__()
#         # ... 定义层 ...
#     def forward(self, x):
#         # x shape: [batch_size, lookback_window, features]
#         # ... 实现前向传播 ...
#         # output shape: [batch_size, prediction_horizon, features]
#         return output
