# --- Main Execution Module ---
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import configuration
from config import (
    DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, SEQ_LEN, PRED_LEN,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, ALPHA_START, ALPHA_END, DEVICE,
    D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
)

# Import modules
from data_preprocessing import load_and_preprocess_data
from models import train_and_predict_linear_teacher, TransformerStudentModel
from training import train_rdt_model
from inference import predict_on_test
from evaluation import evaluate_predictions

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load and Preprocess Data
    X_train, Y_true_train, X_val, Y_true_val, X_test, Y_true_test, scaler, target_col_index, n_features = \
        load_and_preprocess_data(DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, SEQ_LEN, PRED_LEN)

    # 2. Train Linear Teacher and get Soft Labels
    teacher_model, Y_linear_train, Y_linear_val = train_and_predict_linear_teacher(
        X_train, Y_true_train, X_val, PRED_LEN, n_features
    )

    # 3. Initialize Transformer Student Model
    student_model = TransformerStudentModel(
        n_features=n_features,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    print("\nStudent Model Architecture:")
    print(student_model)
    num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # 4. Train the RDT Model
    trained_student_model = train_rdt_model(
        student_model,
        X_train, Y_true_train, Y_linear_train,
        X_val, Y_true_val, Y_linear_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END,
        device=DEVICE
    )

    # 5. Inference on Test Set
    Y_pred_test = predict_on_test(
        trained_student_model, X_test, scaler, target_col_index, BATCH_SIZE, DEVICE
    )

    # 6. Evaluate
    evaluate_predictions(Y_true_test, Y_pred_test, scaler, target_col_index, "RDT Transformer")
    
    # 7. Also evaluate the linear teacher model for comparison
    print("\n=== Evaluating Linear Teacher Model for Comparison ===")
    Y_linear_test_inv = predict_on_test(teacher_model, X_test, scaler, target_col_index, BATCH_SIZE, DEVICE, is_linear=True)
    evaluate_predictions(Y_true_test, Y_linear_test_inv, scaler, target_col_index, "Linear Teacher")

    print("\nRDT Execution Finished.")

if __name__ == "__main__":
    main()