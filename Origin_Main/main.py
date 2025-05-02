# --- Main Execution Module ---
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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

# Define file paths for saving/loading
MODEL_SAVE_PATH = 'results/trained_student_model.pth'
PREDICTIONS_SAVE_PATH = 'results/test_predictions.npy'
LINEAR_PREDICTIONS_SAVE_PATH = 'results/linear_test_predictions.npy' # For linear teacher predictions

def main():
    parser = argparse.ArgumentParser(description='Run RDT Transformer Model.')
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'evaluate_only'],
        help='Run mode: "full" (train and evaluate) or "evaluate_only" (load and evaluate).'
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Running in mode: {args.mode}")
    
    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. Load and Preprocess Data (Required for both modes)
    # We need the scaler and true test labels for evaluation
    print("\nLoading and preprocessing data...")
    X_train, Y_true_train, X_val, Y_true_val, X_test, Y_true_test, scaler, target_col_index, n_features = \
        load_and_preprocess_data(DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, SEQ_LEN, PRED_LEN)
    print("Data loading and preprocessing complete.")
    if args.mode == 'full':
        # --- Full Run: Train and Evaluate ---
        print("\n--- Starting Full Run (Train and Evaluate) ---")
        # 2. Train Linear Teacher and get Soft Labels
        print("\nTraining Linear Teacher...")
        teacher_model, Y_linear_train, Y_linear_val = train_and_predict_linear_teacher(
            X_train, Y_true_train, X_val, PRED_LEN, n_features
        )
        print("Linear Teacher training complete.")
        # 3. Initialize Transformer Student Model
        print("\nInitializing Transformer Student Model...")
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
        print("-" * 30)
        # 4. Train the RDT Model
        print("\nTraining RDT Student Model...")
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
        print("RDT Student Model training complete.")
        # 5. Inference on Test Set for Student Model
        print("\nMaking predictions on test set with RDT Student Model...")
        Y_pred_test = predict_on_test(
            trained_student_model, X_test, scaler, target_col_index, BATCH_SIZE, DEVICE
        )
        print("RDT Student Model prediction complete.")
        # Save the trained student model state dict and predictions
        torch.save(trained_student_model.state_dict(), MODEL_SAVE_PATH)
        np.save(PREDICTIONS_SAVE_PATH, Y_pred_test)
        print(f"Saved trained student model state dict to {MODEL_SAVE_PATH}")
        print(f"Saved RDT test predictions to {PREDICTIONS_SAVE_PATH}")
        # 6. Inference on Test Set for Linear Teacher (for saving comparison predictions)
        print("\nMaking predictions on test set with Linear Teacher...")
        Y_linear_test_pred = predict_on_test(teacher_model, X_test, scaler, target_col_index, BATCH_SIZE, DEVICE, is_linear=True)
        np.save(LINEAR_PREDICTIONS_SAVE_PATH, Y_linear_test_pred)
        print("Linear Teacher prediction complete.")
        print(f"Saved Linear Teacher test predictions to {LINEAR_PREDICTIONS_SAVE_PATH}")
        # 7. Evaluate (will be done regardless of mode after getting predictions)
        print("\n--- Starting Evaluation ---")
    elif args.mode == 'evaluate_only':
        # --- Evaluate Only: Load and Evaluate ---
        print("\n--- Starting Evaluate Only Run (Load and Evaluate) ---")
        # Load saved predictions
        if not os.path.exists(PREDICTIONS_SAVE_PATH) or not os.path.exists(LINEAR_PREDICTIONS_SAVE_PATH):
             print(f"Error: Prediction files not found!")
             print(f"Expected: {PREDICTIONS_SAVE_PATH} and {LINEAR_PREDICTIONS_SAVE_PATH}")
             print("Please run in 'full' mode first to train and save predictions.")
             return # Exit if files not found
        print(f"Loading RDT test predictions from {PREDICTIONS_SAVE_PATH}")
        Y_pred_test = np.load(PREDICTIONS_SAVE_PATH)
        print(f"Loading Linear Teacher test predictions from {LINEAR_PREDICTIONS_SAVE_PATH}")
        Y_linear_test_pred = np.load(LINEAR_PREDICTIONS_SAVE_PATH)
        print("Predictions loaded.")
        # 7. Evaluate (will be done regardless of mode after getting predictions)
        print("\n--- Starting Evaluation ---")
    # --- Evaluation Part (Common to both modes) ---
    # Evaluate RDT Transformer
    print("\n=== Evaluating RDT Transformer Model ===")
    evaluate_predictions(Y_true_test, Y_pred_test, scaler, target_col_index, "RDT Transformer")
    # Evaluate Linear Teacher Model
    print("\n=== Evaluating Linear Teacher Model for Comparison ===")
    # Note: The linear predictions Y_linear_test_pred are already inverse scaled if predict_on_test handles it for linear.
    # Based on your original evaluate call for linear, it seemed predict_on_test for linear *did* inverse scale.
    # If your predict_on_test for linear outputs *scaled* values, you need to inverse transform them here before calling evaluate_predictions.
    # Assuming predict_on_test for linear already outputs inverse scaled:
    evaluate_predictions(Y_true_test, Y_linear_test_pred, scaler, target_col_index, "Linear Teacher")
    print("\nRDT Execution Finished.")
    
if __name__ == "__main__":
    main()