import numpy as np

class BaseAlphaScheduler:
    def __init__(self, alpha_start, alpha_end, total_epochs):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs

    def get_alpha(self, current_epoch):
        raise NotImplementedError

class ConstantScheduler(BaseAlphaScheduler):
    def __init__(self, alpha_value, total_epochs):
        super().__init__(alpha_value, alpha_value, total_epochs) # start 和 end 相同
        self.alpha_value = alpha_value
        print(f"Using Constant Alpha Scheduler with alpha = {self.alpha_value}")

    def get_alpha(self, current_epoch):
        return self.alpha_value

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


def get_alpha_scheduler(cfg):
    """根据配置获取 alpha 调度器实例"""
    schedule_type = cfg.ALPHA_SCHEDULE.lower()
    if schedule_type == 'linear':
        return LinearScheduler(cfg.ALPHA_START, cfg.ALPHA_END, cfg.EPOCHS)
    elif schedule_type == 'exponential':
        # 指数增长通常要求 alpha > 0，如果 alpha_start=0，可能需要调整
        start_alpha = cfg.ALPHA_START if cfg.ALPHA_START > 0 else 1e-6 # 避免 log(0) 或除以 0
        end_alpha = cfg.ALPHA_END
        if end_alpha <= start_alpha and end_alpha > 0: # 处理非增长情况
             print("Warning: ExponentialScheduler with non-increasing alpha, behaving like constant at start.")
             return ConstantScheduler(start_alpha, cfg.EPOCHS)
        elif end_alpha <= 0:
             print("Warning: ExponentialScheduler end alpha <= 0, defaulting to linear.")
             return LinearScheduler(cfg.ALPHA_START, cfg.ALPHA_END, cfg.EPOCHS)

        return ExponentialScheduler(start_alpha, end_alpha, cfg.EPOCHS)
    elif schedule_type == 'constant':
        return ConstantScheduler(cfg.CONSTANT_ALPHA, cfg.EPOCHS)
    else:
        raise ValueError(f"Unsupported alpha schedule type: {schedule_type}")

