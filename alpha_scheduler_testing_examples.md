```markdown
# Testing New Alpha Scheduling Strategies in RDT

This guide provides example command-line arguments for running experiments with each of the new alpha scheduling strategies implemented in `src/schedulers.py`. The entry point for these experiments is `main.py`.

**General Advice for Testing:**

*   **Small Epochs:** Use a small number of epochs (e.g., 10-20) initially to quickly observe the alpha behavior and ensure the experiment setup is correct.
*   **Logging:** Ensure that logging is configured to show alpha values per epoch. The `main.py` script should automatically log the RDT alpha schedule if it's recorded in the training history. Check the `run_*.log` file in the `log` directory and the generated plots for alpha progression.
*   **Dataset/Model Choice:** For dynamic schedulers like `early_stopping_based` and `control_gate`, use datasets and model combinations where you expect to see variations in student/teacher performance or validation loss behavior (e.g., a student model that might overfit or a teacher that provides a clear, but not perfect, signal). This will better test the dynamic adjustment capabilities.
*   **Basic Arguments:** Remember to include other necessary arguments for your experiment, such as:
    *   `--dataset_path data/your_dataset.csv`
    *   `--teacher_model_name DLinear` (or any other model, or `None` for task-only student)
    *   `--student_model_name PatchTST` (or any other model)
    *   `--prediction_horizon 96`
    *   `--lookback_window 192`
    *   `--stability_runs 1` (to keep it quick for testing schedulers)

---

## 1. Fixed Weight Scheduler (`fixed`)

*   **Purpose:** To verify that alpha remains constant throughout training.
*   **Example:**
    ```bash
    python main.py \
        --alpha_schedule fixed \
        --constant_alpha 0.3 \
        --epochs 10 \
        --dataset_path data/national_illness.csv \
        --teacher_model_name DLinear \
        --student_model_name PatchTST \
        --prediction_horizon 96 \
        --lookback_window 192 \
        --stability_runs 1
    ```
*   **Expected Observation:** The alpha value reported in the logs and visible in the alpha schedule plot (if generated) should remain constant at 0.3 for all epochs.

---

## 2. Cosine Annealing Scheduler (`cosine`)

*   **Purpose:** To verify that alpha follows a smooth cosine curve from a starting value to an ending value.
*   **Example:**
    ```bash
    python main.py \
        --alpha_schedule cosine \
        --alpha_start 0.1 \
        --alpha_end 0.9 \
        --epochs 10 \
        --dataset_path data/national_illness.csv \
        --teacher_model_name DLinear \
        --student_model_name PatchTST \
        --prediction_horizon 96 \
        --lookback_window 192 \
        --stability_runs 1
    ```
*   **Expected Observation:** Alpha should start at approximately 0.1 at epoch 0 and smoothly transition to approximately 0.9 by epoch 9, following a cosine annealing curve. This should be visible in the logs and the alpha schedule plot.

---

## 3. Early Stopping Based Scheduler (`early_stopping_based`)

*   **Purpose:** To verify that alpha adjustment occurs based on validation loss stagnation.
*   **Example (Freeze mode):**
    ```bash
    python main.py \
        --alpha_schedule early_stopping_based \
        --alpha_start 0.2 \
        --alpha_end 0.8 \
        --epochs 20 \
        --es_alpha_patience 3 \
        --es_alpha_adjust_mode freeze \
        --dataset_path data/national_illness.csv \
        --teacher_model_name DLinear \
        --student_model_name PatchTST \
        --prediction_horizon 96 \
        --lookback_window 192 \
        --stability_runs 1 \
        --patience 10 # Main model training patience
    ```
*   **Example (Decay to Teacher mode):**
    ```bash
    python main.py \
        --alpha_schedule early_stopping_based \
        --alpha_start 0.5 \
        --alpha_end 0.8 \
        --epochs 20 \
        --es_alpha_patience 3 \
        --es_alpha_adjust_mode decay_to_teacher \
        --es_alpha_adjust_rate 0.05 \
        --dataset_path data/national_illness.csv \
        --teacher_model_name DLinear \
        --student_model_name PatchTST \
        --prediction_horizon 96 \
        --lookback_window 192 \
        --stability_runs 1 \
        --patience 10
    ```
*   **Expected Observation:** Monitor the validation loss (e.g., `val_task_loss` from RDT trainer logs). If the validation loss does not improve for `es_alpha_patience` consecutive epochs, the alpha value should change according to `es_alpha_adjust_mode`.
    *   In `freeze` mode, alpha should become fixed at its value when patience ran out.
    *   In `decay_to_teacher` mode, alpha should decrease by `es_alpha_adjust_rate`.
    *   In `decay_to_student` mode (not shown in example but testable), alpha should increase.
    The logs should explicitly state when the EarlyStoppingBasedScheduler triggers an alpha adjustment.

---

## 4. Control Gate Scheduler (`control_gate`)

*   **Purpose:** To verify dynamic alpha adjustment based on a specified metric (e.g., model/label similarity or error).
*   **Example (Cosine Similarity with Teacher):**
    ```bash
    python main.py \
        --alpha_schedule control_gate \
        --alpha_start 0.2 \
        --alpha_end 0.8 \
        --epochs 15 \
        --control_gate_metric cosine_similarity \
        --control_gate_threshold_low 0.6 \
        --control_gate_threshold_high 0.85 \
        --control_gate_alpha_adjust_rate 0.02 \
        --dataset_path data/national_illness.csv \
        --teacher_model_name DLinear \
        --student_model_name PatchTST \
        --prediction_horizon 96 \
        --lookback_window 192 \
        --stability_runs 1 \
        --patience 10
    ```
*   **Example (MSE Student vs True Labels):**
    ```bash
    python main.py \
        --alpha_schedule control_gate \
        --alpha_start 0.2 \
        --alpha_end 0.8 \
        --epochs 15 \
        --control_gate_metric mse_student_true \
        --control_gate_threshold_low 0.1 \
        --control_gate_threshold_high 0.5 \
        --control_gate_alpha_adjust_rate 0.02 \
        --dataset_path data/national_illness.csv \
        --teacher_model_name DLinear \
        --student_model_name PatchTST \
        --prediction_horizon 96 \
        --lookback_window 192 \
        --stability_runs 1 \
        --patience 10
    ```
*   **Expected Observation:** The logs should show the calculated metric value (e.g., "ControlGateScheduler: Epoch X, Metric (cosine_similarity): Y.YYY, Alpha changed from A.AAA to B.BBB"). Verify that alpha adjusts (increases or decreases by `control_gate_alpha_adjust_rate`) when the metric value crosses the `control_gate_threshold_low` or `control_gate_threshold_high`. If the metric is between the thresholds, alpha should remain unchanged. The initial alpha for ControlGate is typically `(alpha_start + alpha_end) / 2.0`.

---

Remember to adapt file paths, model names, and other hyperparameters as necessary for your specific setup and the dataset being used.
```
