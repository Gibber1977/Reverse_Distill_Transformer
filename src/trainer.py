import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm # 进度条库
from src import config, utils

class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=5, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta # 最小改进量
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss # 分数越高越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best val loss: {self.val_loss_min:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证损失下降时'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        utils.save_model(model, self.path) # 使用 utils 中的保存函数
        self.val_loss_min = val_loss

def get_optimizer(model, cfg):
    """根据配置获取优化器"""
    if cfg.OPTIMIZER.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == 'sgd':
        # SGD 可能需要不同的学习率和 momentum 设置
        return optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")

def get_loss_function(cfg):
    """根据配置获取损失函数"""
    if cfg.LOSS_FN.lower() == 'mse':
        return nn.MSELoss()
    elif cfg.LOSS_FN.lower() == 'mae':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {cfg.LOSS_FN}")


class BaseTrainer:
    """训练器的基类，包含通用训练和验证逻辑"""
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience, model_save_path, model_name="Model"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path)
        self.history = {'train_loss': [], 'val_loss': []}

    def _train_epoch(self):
        raise NotImplementedError

    def _validate_epoch(self):
        raise NotImplementedError

    def train(self):
        print(f"\n--- Starting Training for {self.model_name} ---")
        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # --- 训练 ---
            train_loss = self._train_epoch()
            self.history['train_loss'].append(train_loss)

            # --- 验证 ---
            val_loss = self._validate_epoch()
            self.history['val_loss'].append(val_loss)

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_duration:.2f}s")

            # --- 早停检查 ---
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        total_time = time.time() - start_time
        print(f"--- Training Finished for {self.model_name} ---")
        print(f"Total Training Time: {total_time:.2f}s")
        print(f"Best validation loss achieved: {self.early_stopping.val_loss_min:.6f}")
        print(f"Model saved to: {self.model_save_path}")

        # 加载性能最好的模型
        self.model = utils.load_model(self.model, self.model_save_path, self.device)

        return self.model, self.history


class StandardTrainer(BaseTrainer):
    """标准训练流程 (用于教师模型或基线学生模型)"""
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience, model_save_path, model_name="StandardModel"):
        super().__init__(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience, model_save_path, model_name)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        # 使用 tqdm 创建进度条
        train_iterator = tqdm(self.train_loader, desc=f"Epoch {len(self.history['train_loss']) + 1}/{self.epochs} Training", leave=False)
        for batch_x, batch_y in train_iterator:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()

            input_dict = {'insample_y': batch_x}
            outputs = self.model(input_dict) # [batch, horizon, features]

            loss = self.loss_fn(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item()) # 在进度条上显示当前 batch loss

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        val_iterator = tqdm(self.val_loader, desc=f"Epoch {len(self.history['val_loss']) + 1}/{self.epochs} Validation", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in val_iterator:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                input_dict = {'insample_y': batch_x}
                outputs = self.model(input_dict)

                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()
                val_iterator.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)


class RDT_Trainer(BaseTrainer):
    """RDT 训练流程"""
    def __init__(self, student_model, teacher_model, train_loader, val_loader, optimizer, task_loss_fn, distill_loss_fn, alpha_scheduler, device, epochs, patience, model_save_path, model_name="RDT_Student"):
        super().__init__(student_model, train_loader, val_loader, optimizer, task_loss_fn, device, epochs, patience, model_save_path, model_name) # loss_fn 作为 task_loss_fn
        self.teacher_model = teacher_model.to(device).eval() # 教师模型设为评估模式且不训练
        self.distill_loss_fn = distill_loss_fn
        self.alpha_scheduler = alpha_scheduler
        self.history['alpha'] = [] # 记录 alpha 变化

    def _train_epoch(self):
        self.model.train() # student model 设置为训练模式
        self.teacher_model.eval() # teacher model 保持评估模式
        total_loss = 0
        total_task_loss = 0
        total_distill_loss = 0
        current_epoch = len(self.history['train_loss']) # 当前 epoch (从 0 开始)
        alpha = self.alpha_scheduler.get_alpha(current_epoch)
        self.history['alpha'].append(alpha) # 记录当前 alpha

        train_iterator = tqdm(self.train_loader, desc=f"Epoch {current_epoch + 1}/{self.epochs} RDT Training (alpha={alpha:.3f})", leave=False)

        for batch_x, batch_y_true in train_iterator:
            batch_x, batch_y_true = batch_x.to(self.device), batch_y_true.to(self.device)

            # 1. 教师模型预测 (不计算梯度)
            with torch.no_grad():
                input_dict_teacher = {'insample_y': batch_x}
                batch_y_teacher = self.teacher_model(input_dict_teacher)

            # 2. 学生模型预测
            self.optimizer.zero_grad()
            input_dict_student = {'insample_y': batch_x}
            batch_y_student = self.model(input_dict_student) # self.model is student

            # 3. 计算损失
            loss_task = self.loss_fn(batch_y_student, batch_y_true)
            # detach教师预测，阻止梯度流向教师
            loss_distill = self.distill_loss_fn(batch_y_student, batch_y_teacher.detach())

            # 4. 计算总损失
            loss_total = alpha * loss_task + (1 - alpha) * loss_distill

            # 5. 反向传播和优化 (只更新学生模型)
            loss_total.backward()
            self.optimizer.step()

            total_loss += loss_total.item()
            total_task_loss += loss_task.item()
            total_distill_loss += loss_distill.item()
            train_iterator.set_postfix(loss=loss_total.item(), task_loss=loss_task.item(), dist_loss=loss_distill.item())

        avg_total_loss = total_loss / len(self.train_loader)
        avg_task_loss = total_task_loss / len(self.train_loader)
        avg_distill_loss = total_distill_loss / len(self.train_loader)
        print(f"Epoch {current_epoch+1} Avg Train Losses: Total={avg_total_loss:.6f}, Task={avg_task_loss:.6f}, Distill={avg_distill_loss:.6f}")

        return avg_total_loss # 返回总损失用于记录，但早停基于验证集 Task Loss

    def _validate_epoch(self):
        # RDT 的验证通常只关心学生模型在真实任务上的表现 (Task Loss)
        self.model.eval()
        total_task_loss = 0
        val_iterator = tqdm(self.val_loader, desc=f"Epoch {len(self.history['val_loss']) + 1}/{self.epochs} RDT Validation", leave=False)
        with torch.no_grad():
            for batch_x, batch_y_true in val_iterator:
                batch_x, batch_y_true = batch_x.to(self.device), batch_y_true.to(self.device)
                
                input_dict = {'insample_y': batch_x}
                batch_y_student = self.model(input_dict)
                
                loss_task = self.loss_fn(batch_y_student, batch_y_true) # 只计算 Task Loss
                total_task_loss += loss_task.item()
                val_iterator.set_postfix(val_task_loss=loss_task.item())

        avg_val_loss = total_task_loss / len(self.val_loader)
        # 注意：早停是基于这个验证 Task Loss
        return avg_val_loss
