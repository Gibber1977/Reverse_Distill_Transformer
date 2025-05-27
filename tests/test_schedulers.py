import unittest
import torch
import numpy as np
from src.schedulers import (
    BaseAlphaScheduler,
    ConstantScheduler,
    FixedWeightScheduler,
    LinearScheduler,
    CosineAnnealingScheduler,
    ExponentialScheduler,
    EarlyStoppingBasedScheduler,
    ControlGateScheduler,
    get_alpha_scheduler
)

# Mock Config class to simulate the main Config object
class MockConfig:
    def __init__(self):
        # General
        self.EPOCHS = 10
        self.ALPHA_START = 0.1
        self.ALPHA_END = 0.9
        self.CONSTANT_ALPHA = 0.5
        self.ALPHA_SCHEDULE = 'linear' # Default, can be changed per test

        # EarlyStoppingBasedScheduler
        self.ES_ALPHA_PATIENCE = 3
        self.ES_ALPHA_ADJUST_MODE = 'freeze' # 'freeze', 'decay_to_teacher', 'decay_to_student'
        self.ES_ALPHA_ADJUST_RATE = 0.01

        # ControlGateScheduler
        self.CONTROL_GATE_METRIC = 'cosine_similarity' # 'mse_student_true', 'mse_student_teacher'
        self.CONTROL_GATE_THRESHOLD_LOW = 0.5
        self.CONTROL_GATE_THRESHOLD_HIGH = 0.8
        self.CONTROL_GATE_ALPHA_ADJUST_RATE = 0.05
        self.CONTROL_GATE_TARGET_SIMILARITY = 0.7 # Optional
        self.CONTROL_GATE_MSE_STUDENT_TARGET = 0.1 # Optional


class TestFixedWeightScheduler(unittest.TestCase):
    def test_initialization_and_get_alpha(self):
        cfg = MockConfig()
        cfg.CONSTANT_ALPHA = 0.6
        scheduler = FixedWeightScheduler(alpha_value=cfg.CONSTANT_ALPHA, total_epochs=cfg.EPOCHS)
        self.assertEqual(scheduler.alpha_value, 0.6)
        for epoch in range(cfg.EPOCHS):
            self.assertEqual(scheduler.get_alpha(epoch), 0.6)

class TestCosineAnnealingScheduler(unittest.TestCase):
    def test_alpha_progression(self):
        cfg = MockConfig()
        cfg.ALPHA_START = 0.0
        cfg.ALPHA_END = 1.0
        cfg.EPOCHS = 11 # Use 11 epochs for 0 to 10, easier mid-point
        scheduler = CosineAnnealingScheduler(alpha_start=cfg.ALPHA_START, alpha_end=cfg.ALPHA_END, total_epochs=cfg.EPOCHS)

        # Alpha at start (epoch 0) should be alpha_start
        self.assertAlmostEqual(scheduler.get_alpha(0), cfg.ALPHA_START, places=6)

        # Alpha at mid (epoch 5 for 11 total epochs, progress = 0.5)
        # cos(pi * 0.5) = 0. So, alpha = alpha_end + 0.5 * (alpha_start - alpha_end) * (1 + 0)
        # alpha = alpha_end + 0.5 * alpha_start - 0.5 * alpha_end = 0.5 * (alpha_start + alpha_end)
        expected_mid_alpha = cfg.ALPHA_END + 0.5 * (cfg.ALPHA_START - cfg.ALPHA_END) * (1 + np.cos(np.pi * 0.5))
        self.assertAlmostEqual(scheduler.get_alpha(cfg.EPOCHS // 2), expected_mid_alpha, places=6)


        # Alpha at end (epoch 10) should be alpha_end
        self.assertAlmostEqual(scheduler.get_alpha(cfg.EPOCHS - 1), cfg.ALPHA_END, places=6)

    def test_single_epoch(self):
        scheduler = CosineAnnealingScheduler(alpha_start=0.1, alpha_end=0.9, total_epochs=1)
        self.assertAlmostEqual(scheduler.get_alpha(0), 0.9, places=6)


class TestEarlyStoppingBasedScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.total_epochs = self.cfg.EPOCHS

    def test_initialization(self):
        scheduler = EarlyStoppingBasedScheduler(self.cfg, self.total_epochs)
        self.assertEqual(scheduler.current_alpha, self.cfg.ALPHA_START)
        self.assertEqual(scheduler.es_alpha_patience, self.cfg.ES_ALPHA_PATIENCE)
        self.assertEqual(scheduler.es_alpha_adjust_mode, self.cfg.ES_ALPHA_ADJUST_MODE)
        self.assertEqual(scheduler.es_alpha_adjust_rate, self.cfg.ES_ALPHA_ADJUST_RATE)
        self.assertEqual(scheduler.patience_counter, 0)
        self.assertEqual(scheduler.best_val_loss, float('inf'))
        self.assertFalse(scheduler.frozen)

    def test_val_loss_improves(self):
        scheduler = EarlyStoppingBasedScheduler(self.cfg, self.total_epochs)
        initial_alpha = scheduler.get_alpha(0)
        val_loss = 1.0
        for epoch in range(self.cfg.ES_ALPHA_PATIENCE + 1):
            val_loss *= 0.9 # Consistent improvement
            scheduler.update(epoch, val_loss=val_loss)
            self.assertEqual(scheduler.get_alpha(epoch), initial_alpha)
            self.assertEqual(scheduler.patience_counter, 0)
        self.assertFalse(scheduler.frozen)

    def test_patience_increments(self):
        scheduler = EarlyStoppingBasedScheduler(self.cfg, self.total_epochs)
        scheduler.update(0, val_loss=1.0) # Initial best loss
        for epoch in range(1, self.cfg.ES_ALPHA_PATIENCE):
            scheduler.update(epoch, val_loss=1.1) # No improvement
            self.assertEqual(scheduler.patience_counter, epoch)
            self.assertEqual(scheduler.get_alpha(epoch), self.cfg.ALPHA_START)
        self.assertFalse(scheduler.frozen)

    def test_patience_runs_out_freeze_mode(self):
        self.cfg.ES_ALPHA_ADJUST_MODE = 'freeze'
        scheduler = EarlyStoppingBasedScheduler(self.cfg, self.total_epochs)
        scheduler.update(0, val_loss=1.0)
        for epoch in range(1, self.cfg.ES_ALPHA_PATIENCE):
            scheduler.update(epoch, val_loss=1.1) # No improvement

        # Trigger adjustment
        scheduler.update(self.cfg.ES_ALPHA_PATIENCE, val_loss=1.1)
        self.assertTrue(scheduler.frozen)
        alpha_after_freeze = scheduler.get_alpha(self.cfg.ES_ALPHA_PATIENCE)
        self.assertEqual(alpha_after_freeze, self.cfg.ALPHA_START)
        # Further updates should not change alpha
        scheduler.update(self.cfg.ES_ALPHA_PATIENCE + 1, val_loss=0.5) # Improvement
        self.assertEqual(scheduler.get_alpha(self.cfg.ES_ALPHA_PATIENCE+1), alpha_after_freeze)

    def test_patience_runs_out_decay_to_teacher_mode(self):
        self.cfg.ES_ALPHA_ADJUST_MODE = 'decay_to_teacher'
        self.cfg.ALPHA_START = 0.5 # Start somewhere in the middle
        self.cfg.ES_ALPHA_ADJUST_RATE = 0.1
        scheduler = EarlyStoppingBasedScheduler(self.cfg, self.total_epochs)
        initial_alpha = scheduler.current_alpha

        scheduler.update(0, val_loss=1.0)
        for epoch in range(1, self.cfg.ES_ALPHA_PATIENCE +1 ): # Reach patience trigger
            scheduler.update(epoch, val_loss=1.1)

        expected_alpha = initial_alpha - self.cfg.ES_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(self.cfg.ES_ALPHA_PATIENCE), expected_alpha, places=6)
        self.assertEqual(scheduler.patience_counter, 0) # Resets after adjustment

        # Test boundary (not going below 0)
        scheduler.current_alpha = 0.05 # Set alpha close to 0
        scheduler.best_val_loss = 0.5 # Reset best loss
        scheduler.update(0, val_loss=0.5)
        for epoch in range(1, self.cfg.ES_ALPHA_PATIENCE + 1):
            scheduler.update(epoch, val_loss=0.6)
        self.assertAlmostEqual(scheduler.get_alpha(self.cfg.ES_ALPHA_PATIENCE), 0.0, places=6)


    def test_patience_runs_out_decay_to_student_mode(self):
        self.cfg.ES_ALPHA_ADJUST_MODE = 'decay_to_student'
        self.cfg.ALPHA_START = 0.5 # Start somewhere in the middle
        self.cfg.ES_ALPHA_ADJUST_RATE = 0.1
        scheduler = EarlyStoppingBasedScheduler(self.cfg, self.total_epochs)
        initial_alpha = scheduler.current_alpha

        scheduler.update(0, val_loss=1.0)
        for epoch in range(1, self.cfg.ES_ALPHA_PATIENCE +1 ): # Reach patience trigger
            scheduler.update(epoch, val_loss=1.1)

        expected_alpha = initial_alpha + self.cfg.ES_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(self.cfg.ES_ALPHA_PATIENCE), expected_alpha, places=6)

        # Test boundary (not going above 1)
        scheduler.current_alpha = 0.95 # Set alpha close to 1
        scheduler.best_val_loss = 0.5 # Reset best loss
        scheduler.update(0, val_loss=0.5)
        for epoch in range(1, self.cfg.ES_ALPHA_PATIENCE + 1):
            scheduler.update(epoch, val_loss=0.6)
        self.assertAlmostEqual(scheduler.get_alpha(self.cfg.ES_ALPHA_PATIENCE), 1.0, places=6)


class TestControlGateScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.total_epochs = self.cfg.EPOCHS
        # Dummy tensors (batch_size, sequence_length, num_features)
        self.student_preds = torch.randn(16, 20, 5)
        self.teacher_preds = torch.randn(16, 20, 5)
        self.true_labels = torch.randn(16, 20, 5)
        # Initial alpha is usually (alpha_start + alpha_end) / 2
        self.initial_alpha = (self.cfg.ALPHA_START + self.cfg.ALPHA_END) / 2.0

    def test_initialization(self):
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)
        self.assertAlmostEqual(scheduler.current_alpha, self.initial_alpha, places=6)
        self.assertEqual(scheduler.metric_name, self.cfg.CONTROL_GATE_METRIC)
        self.assertEqual(scheduler.threshold_low, self.cfg.CONTROL_GATE_THRESHOLD_LOW)
        self.assertEqual(scheduler.threshold_high, self.cfg.CONTROL_GATE_THRESHOLD_HIGH)
        self.assertEqual(scheduler.adjust_rate, self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE)

    def test_metric_cosine_similarity(self):
        self.cfg.CONTROL_GATE_METRIC = 'cosine_similarity'
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)
        self.assertEqual(scheduler.metric_name, 'cosine_similarity')

        # Scenario 1: Metric below threshold_low (alpha decreases)
        # Mock _calculate_metric to return a low similarity
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_LOW - 0.1
        scheduler.update(0, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        expected_alpha = self.initial_alpha - self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(0), expected_alpha, places=6)

        # Scenario 2: Metric above threshold_high (alpha increases)
        scheduler.current_alpha = self.initial_alpha # Reset alpha
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_HIGH + 0.1
        scheduler.update(1, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        expected_alpha = self.initial_alpha + self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(1), expected_alpha, places=6)

        # Scenario 3: Metric between thresholds (alpha unchanged)
        scheduler.current_alpha = self.initial_alpha # Reset alpha
        scheduler._calculate_metric = lambda sp, tp, tl: (self.cfg.CONTROL_GATE_THRESHOLD_LOW + self.cfg.CONTROL_GATE_THRESHOLD_HIGH) / 2
        scheduler.update(2, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        self.assertAlmostEqual(scheduler.get_alpha(2), self.initial_alpha, places=6)

    def test_metric_mse_student_true(self):
        self.cfg.CONTROL_GATE_METRIC = 'mse_student_true'
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)
        self.assertEqual(scheduler.metric_name, 'mse_student_true')

        # Scenario 1: MSE above threshold_high (bad MSE, alpha decreases)
        scheduler.current_alpha = self.initial_alpha # Reset alpha
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_HIGH + 0.1
        scheduler.update(0, student_preds=self.student_preds, true_labels=self.true_labels)
        expected_alpha = self.initial_alpha - self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(0), expected_alpha, places=6)

        # Scenario 2: MSE below threshold_low (good MSE, alpha increases)
        scheduler.current_alpha = self.initial_alpha # Reset alpha
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_LOW - 0.1
        scheduler.update(1, student_preds=self.student_preds, true_labels=self.true_labels)
        expected_alpha = self.initial_alpha + self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(1), expected_alpha, places=6)

    def test_metric_mse_student_teacher(self):
        self.cfg.CONTROL_GATE_METRIC = 'mse_student_teacher'
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)
        self.assertEqual(scheduler.metric_name, 'mse_student_teacher')

        # Scenario 1: MSE above threshold_high (bad MSE, alpha decreases)
        scheduler.current_alpha = self.initial_alpha # Reset alpha
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_HIGH + 0.1
        scheduler.update(0, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        expected_alpha = self.initial_alpha - self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(0), expected_alpha, places=6)

        # Scenario 2: MSE below threshold_low (good MSE, alpha increases)
        scheduler.current_alpha = self.initial_alpha # Reset alpha
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_LOW - 0.1
        scheduler.update(1, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        expected_alpha = self.initial_alpha + self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE
        self.assertAlmostEqual(scheduler.get_alpha(1), expected_alpha, places=6)

    def test_alpha_boundaries(self):
        self.cfg.CONTROL_GATE_METRIC = 'cosine_similarity'
        self.cfg.CONTROL_GATE_ALPHA_ADJUST_RATE = 0.6 # Large rate for easier boundary check
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)
        
        # Test not going below 0
        scheduler.current_alpha = 0.1 # Start close to 0
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_LOW - 0.1 # Decrease alpha
        scheduler.update(0, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        self.assertAlmostEqual(scheduler.get_alpha(0), 0.0, places=6)

        # Test not going above 1
        scheduler.current_alpha = 0.9 # Start close to 1
        scheduler._calculate_metric = lambda sp, tp, tl: self.cfg.CONTROL_GATE_THRESHOLD_HIGH + 0.1 # Increase alpha
        scheduler.update(1, student_preds=self.student_preds, teacher_preds=self.teacher_preds)
        self.assertAlmostEqual(scheduler.get_alpha(1), 1.0, places=6)

    def test_real_metric_calculation_cosine(self):
        # Test with actual cosine similarity calculation
        self.cfg.CONTROL_GATE_METRIC = 'cosine_similarity'
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)
        
        # Student and teacher are identical
        preds = torch.ones_like(self.student_preds)
        metric_val = scheduler._calculate_metric(preds, preds, self.true_labels)
        self.assertAlmostEqual(metric_val, 1.0, places=5) # Cosine sim should be 1

        # Student and teacher are orthogonal (on average for random)
        # This is harder to guarantee orthogonality for random tensors to be exactly 0.
        # Instead, we'll check if the update logic is called.
        # We can create specific vectors for this if needed.
        s_preds = torch.tensor([[[1.0, 0.0]]]).float()
        t_preds = torch.tensor([[[0.0, 1.0]]]).float()
        metric_val = scheduler._calculate_metric(s_preds, t_preds, None)
        self.assertAlmostEqual(metric_val, 0.0, places=5) # Cosine sim should be 0

    def test_real_metric_calculation_mse(self):
        self.cfg.CONTROL_GATE_METRIC = 'mse_student_true'
        scheduler = ControlGateScheduler(self.cfg, self.total_epochs)

        # Student and true are identical
        preds = torch.ones_like(self.student_preds)
        metric_val = scheduler._calculate_metric(preds, self.teacher_preds, preds) # teacher_preds not used here
        self.assertAlmostEqual(metric_val, 0.0, places=5) # MSE should be 0

        # Student and true are different
        s_preds = torch.ones_like(self.student_preds)
        t_labels = torch.zeros_like(self.student_preds)
        metric_val = scheduler._calculate_metric(s_preds, self.teacher_preds, t_labels)
        self.assertAlmostEqual(metric_val, 1.0, places=5) # MSE should be 1 if preds are 1 and labels are 0


class TestGetAlphaScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()

    def test_get_linear(self):
        self.cfg.ALPHA_SCHEDULE = 'linear'
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, LinearScheduler)

    def test_get_cosine(self):
        self.cfg.ALPHA_SCHEDULE = 'cosine'
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, CosineAnnealingScheduler)

    def test_get_exponential(self):
        self.cfg.ALPHA_SCHEDULE = 'exponential'
        self.cfg.ALPHA_START = 0.1 # Ensure > 0 for exponential
        self.cfg.ALPHA_END = 0.9
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, ExponentialScheduler)

    def test_get_fixed(self):
        self.cfg.ALPHA_SCHEDULE = 'fixed'
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, FixedWeightScheduler)
        self.assertEqual(scheduler.alpha_value, self.cfg.CONSTANT_ALPHA)

    def test_get_constant(self): # 'constant' is an alias for 'fixed'
        self.cfg.ALPHA_SCHEDULE = 'constant'
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, FixedWeightScheduler) # Currently FixedWeightScheduler is returned
        self.assertEqual(scheduler.alpha_value, self.cfg.CONSTANT_ALPHA)


    def test_get_early_stopping(self):
        self.cfg.ALPHA_SCHEDULE = 'early_stopping_based'
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, EarlyStoppingBasedScheduler)

    def test_get_control_gate(self):
        self.cfg.ALPHA_SCHEDULE = 'control_gate'
        scheduler = get_alpha_scheduler(self.cfg)
        self.assertIsInstance(scheduler, ControlGateScheduler)

    def test_get_invalid(self):
        self.cfg.ALPHA_SCHEDULE = 'invalid_scheduler_type'
        with self.assertRaises(ValueError):
            get_alpha_scheduler(self.cfg)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
