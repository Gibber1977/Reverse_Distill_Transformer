import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error as mse_sklearn # For ControlGateScheduler if needed
# Attempt to import project-specific cosine_similarity, fall back if not found
try:
    from src.evaluator import cosine_similarity as cosine_similarity_project
except ImportError:
    cosine_similarity_project = None
    print("Warning: src.evaluator.cosine_similarity not found. ControlGateScheduler using 'cosine_similarity' might fail if not using PyTorch's version.")


class BaseAlphaScheduler:
    def __init__(self, alpha_start, alpha_end, total_epochs):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs

    def get_alpha(self, current_epoch):
        raise NotImplementedError

    def update(self, current_epoch, val_loss=None, student_preds=None, teacher_preds=None, true_labels=None, student_model=None):
        """
        根据验证集信息（如验证损失、模型预测）动态调整 alpha。
        此方法在基类中默认不实现，由需要动态调整的子类实现。
        参数:
            current_epoch (int): 当前的 epoch 编号。
            val_loss (float, optional): 当前 epoch 的验证损失。
            student_preds (torch.Tensor, optional): 学生模型在验证集上的预测结果。
            teacher_preds (torch.Tensor, optional): 教师模型在验证集上的预测结果。
            true_labels (torch.Tensor, optional): 验证集上的真实标签。
            student_model (torch.nn.Module, optional): 学生模型实例，用于某些调度器。
        """
        pass # 默认不执行任何操作

class ConstantScheduler(BaseAlphaScheduler):
    def __init__(self, alpha_value, total_epochs):
        super().__init__(alpha_value, alpha_value, total_epochs) # start 和 end 相同
        self.alpha_value = alpha_value
        print(f"Using Constant Alpha Scheduler with alpha = {self.alpha_value}")

    def get_alpha(self, current_epoch):
        return self.alpha_value

class FixedWeightScheduler(ConstantScheduler):
    def __init__(self, alpha_value, total_epochs):
        super().__init__(alpha_value, total_epochs)
        print(f"Using Fixed Weight Alpha Scheduler with alpha = {self.alpha_value}")

class LinearScheduler(BaseAlphaScheduler):
    def __init__(self, alpha_start, alpha_end, total_epochs):
        super().__init__(alpha_start, alpha_end, total_epochs)
        print(f"Using Linear Alpha Scheduler: start={alpha_start}, end={alpha_end}, epochs={total_epochs}")

    def get_alpha(self, current_epoch):
        if self.total_epochs <= 1:
            return self.alpha_end
        # 从 0 开始计数 epoch
        progress = min(current_epoch / (self.total_epochs - 1), 1.0)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        return alpha

    def update(self, current_epoch, val_loss=None, student_preds=None, teacher_preds=None, true_labels=None, student_model=None):
        # 对于线性调度器，update 方法不执行任何操作
        pass

class CosineAnnealingScheduler(BaseAlphaScheduler):
    def __init__(self, alpha_start, alpha_end, total_epochs):
        super().__init__(alpha_start, alpha_end, total_epochs)
        print(f"Using Cosine Annealing Alpha Scheduler: start={alpha_start}, end={alpha_end}, epochs={total_epochs}")

    def get_alpha(self, current_epoch):
        if self.total_epochs <= 1:
            return self.alpha_end
        progress = min(current_epoch / (self.total_epochs - 1), 1.0)
        # 使用余弦退火公式
        alpha = self.alpha_end + 0.5 * (self.alpha_start - self.alpha_end) * (1 + np.cos(np.pi * progress))
        return alpha

class ExponentialScheduler(BaseAlphaScheduler):
    def __init__(self, alpha_start, alpha_end, total_epochs):
        super().__init__(alpha_start, alpha_end, total_epochs)
        if alpha_start <= 0 or alpha_end <= 0:
            raise ValueError("ExponentialScheduler requires positive start and end alpha values.")
        # 计算增长率 r: alpha_end = alpha_start * (r ^ (total_epochs - 1))
        if self.total_epochs > 1:
             # 防止 alpha_start 和 alpha_end 非常接近或相等时出现计算问题
            if abs(alpha_end - alpha_start) < 1e-9:
                self.rate = 1.0
            else:
                self.rate = (alpha_end / alpha_start) ** (1.0 / (total_epochs - 1))
        else:
             self.rate = 1.0 # 如果只有一个 epoch，alpha 直接是 alpha_end
        print(f"Using Exponential Alpha Scheduler: start={alpha_start}, end={alpha_end}, epochs={total_epochs}, rate={self.rate:.4f}")


    def get_alpha(self, current_epoch):
        if self.total_epochs <= 1:
            return self.alpha_end
        alpha = self.alpha_start * (self.rate ** current_epoch)
        # 限制 alpha 不超过 alpha_end (防止浮点误差导致略微超出)
        return min(alpha, self.alpha_end) if self.rate > 1.0 else max(alpha, self.alpha_end)

    def update(self, current_epoch, val_loss=None, student_preds=None, teacher_preds=None, true_labels=None, student_model=None):
        # 对于指数调度器，update 方法不执行任何操作
        pass

class EarlyStoppingBasedScheduler(BaseAlphaScheduler):
    def __init__(self, config, total_epochs):
        super().__init__(config.ALPHA_START, config.ALPHA_END, total_epochs)
        self.es_alpha_patience = config.ES_ALPHA_PATIENCE
        self.es_alpha_adjust_mode = config.ES_ALPHA_ADJUST_MODE.lower()
        self.es_alpha_adjust_rate = config.ES_ALPHA_ADJUST_RATE
        
        self.current_alpha = config.ALPHA_START # Initial alpha
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.frozen = False # Flag to indicate if alpha is frozen

        print(f"Using Early Stopping Based Alpha Scheduler: initial_alpha={self.current_alpha}, patience={self.es_alpha_patience}, mode='{self.es_alpha_adjust_mode}', rate={self.es_alpha_adjust_rate}")

    def get_alpha(self, current_epoch):
        # Optionally, could have an underlying linear/cosine progression here if not frozen
        # For now, it's either the initial alpha or the adjusted alpha
        return self.current_alpha

    def update(self, current_epoch, val_loss=None, student_preds=None, teacher_preds=None, true_labels=None, student_model=None):
        if self.frozen:
            return

        if val_loss is None:
            print("Warning: EarlyStoppingBasedScheduler requires val_loss for updates.")
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.es_alpha_patience:
            print(f"EarlyStoppingBasedScheduler: Patience triggered at epoch {current_epoch}. Adjusting alpha.")
            if self.es_alpha_adjust_mode == 'freeze':
                self.frozen = True
                print(f"Alpha frozen at {self.current_alpha}")
            elif self.es_alpha_adjust_mode == 'decay_to_teacher': # Trust teacher more
                self.current_alpha = max(0.0, self.current_alpha - self.es_alpha_adjust_rate)
                print(f"Alpha decayed towards teacher, new alpha: {self.current_alpha}")
            elif self.es_alpha_adjust_mode == 'decay_to_student': # Trust student more
                self.current_alpha = min(1.0, self.current_alpha + self.es_alpha_adjust_rate)
                print(f"Alpha decayed towards student, new alpha: {self.current_alpha}")
            else:
                print(f"Warning: Unknown ES_ALPHA_ADJUST_MODE '{self.es_alpha_adjust_mode}'. Alpha not changed.")
            self.patience_counter = 0 # Reset patience after adjustment

class ControlGateScheduler(BaseAlphaScheduler):
    def __init__(self, config, total_epochs):
        super().__init__(config.ALPHA_START, config.ALPHA_END, total_epochs)
        self.metric_name = config.CONTROL_GATE_METRIC.lower()
        self.threshold_low = config.CONTROL_GATE_THRESHOLD_LOW
        self.threshold_high = config.CONTROL_GATE_THRESHOLD_HIGH
        self.adjust_rate = config.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.target_similarity = config.CONTROL_GATE_TARGET_SIMILARITY # Optional
        self.target_mse_student = config.CONTROL_GATE_MSE_STUDENT_TARGET # Optional

        # Initialize alpha: could be start, end, or average
        self.current_alpha = (config.ALPHA_START + config.ALPHA_END) / 2.0 
        print(f"Using Control Gate Alpha Scheduler: initial_alpha={self.current_alpha}, metric='{self.metric_name}', low_thresh={self.threshold_low}, high_thresh={self.threshold_high}, rate={self.adjust_rate}")

    def get_alpha(self, current_epoch):
        return self.current_alpha

    def _calculate_metric(self, student_preds, teacher_preds, true_labels):
        if student_preds is None: return None # Cannot calculate if student preds are missing

        # Ensure tensors are on CPU and converted to numpy if using sklearn/numpy functions
        # For PyTorch functions, ensure they are on the same device
        
        # Flatten predictions for metric calculation (batch_size * horizon, num_features)
        # Assuming student_preds, teacher_preds, true_labels are [batch, horizon, features]
        s_preds_flat = student_preds.reshape(-1, student_preds.shape[-1])
        
        if self.metric_name == 'cosine_similarity':
            if teacher_preds is None: return None
            t_preds_flat = teacher_preds.reshape(-1, teacher_preds.shape[-1])
            # Using torch.nn.functional.cosine_similarity
            # Ensure inputs are float
            sim = F.cosine_similarity(s_preds_flat.float(), t_preds_flat.float(), dim=-1).mean().item()
            return sim
        elif self.metric_name == 'mse_student_true':
            if true_labels is None: return None
            true_labels_flat = true_labels.reshape(-1, true_labels.shape[-1])
            mse = F.mse_loss(s_preds_flat.float(), true_labels_flat.float()).item()
            return mse
        elif self.metric_name == 'mse_student_teacher':
            if teacher_preds is None: return None
            t_preds_flat = teacher_preds.reshape(-1, teacher_preds.shape[-1])
            mse = F.mse_loss(s_preds_flat.float(), t_preds_flat.float()).item()
            return mse
        else:
            print(f"Warning: Unknown ControlGate metric '{self.metric_name}'.")
            return None

    def update(self, current_epoch, val_loss=None, student_preds=None, teacher_preds=None, true_labels=None, student_model=None):
        metric_val = self._calculate_metric(student_preds, teacher_preds, true_labels)

        if metric_val is None:
            print(f"ControlGateScheduler: Metric calculation failed or inputs missing at epoch {current_epoch}. Alpha not changed.")
            return

        prev_alpha = self.current_alpha
        if self.metric_name == 'cosine_similarity': # Higher is better
            if metric_val < self.threshold_low: # Student is too different from teacher
                self.current_alpha = max(0.0, self.current_alpha - self.adjust_rate) # Trust teacher more
            elif metric_val > self.threshold_high: # Student is very similar to teacher
                self.current_alpha = min(1.0, self.current_alpha + self.adjust_rate) # Trust student more (task loss)
        elif 'mse' in self.metric_name: # Lower is better
            if metric_val > self.threshold_high: # Error is too high
                self.current_alpha = max(0.0, self.current_alpha - self.adjust_rate) # Trust teacher more
            elif metric_val < self.threshold_low: # Error is very low
                self.current_alpha = min(1.0, self.current_alpha + self.adjust_rate) # Trust student/task more
        
        if prev_alpha != self.current_alpha:
            print(f"ControlGateScheduler: Epoch {current_epoch}, Metric ({self.metric_name}): {metric_val:.4f}, Alpha changed from {prev_alpha:.4f} to {self.current_alpha:.4f}")
        else:
            print(f"ControlGateScheduler: Epoch {current_epoch}, Metric ({self.metric_name}): {metric_val:.4f}, Alpha remains {self.current_alpha:.4f}")


def get_alpha_scheduler(cfg):
    """根据配置获取 alpha 调度器实例"""
    schedule_type = cfg.ALPHA_SCHEDULE.lower()
    if schedule_type == 'linear':
        return LinearScheduler(cfg.ALPHA_START, cfg.ALPHA_END, cfg.EPOCHS)
    elif schedule_type == 'cosine':
        return CosineAnnealingScheduler(cfg.ALPHA_START, cfg.ALPHA_END, cfg.EPOCHS)
    elif schedule_type == 'exponential':
        start_alpha = cfg.ALPHA_START if cfg.ALPHA_START > 0 else 1e-6 
        end_alpha = cfg.ALPHA_END
        if end_alpha <= start_alpha and end_alpha > 0:
             print("Warning: ExponentialScheduler with non-increasing alpha, behaving like constant at start.")
             return ConstantScheduler(start_alpha, cfg.EPOCHS) # Or FixedWeightScheduler
        elif end_alpha <= 0:
             print("Warning: ExponentialScheduler end alpha <= 0, defaulting to linear.")
             return LinearScheduler(cfg.ALPHA_START, cfg.ALPHA_END, cfg.EPOCHS)
        return ExponentialScheduler(start_alpha, end_alpha, cfg.EPOCHS)
    elif schedule_type == 'constant' or schedule_type == 'fixed': # Added 'fixed'
        return FixedWeightScheduler(cfg.CONSTANT_ALPHA, cfg.EPOCHS)
    elif schedule_type == 'early_stopping_based':
        return EarlyStoppingBasedScheduler(cfg, cfg.EPOCHS)
    elif schedule_type == 'control_gate':
        return ControlGateScheduler(cfg, cfg.EPOCHS)
    else:
        raise ValueError(f"Unsupported alpha schedule type: {schedule_type}")

