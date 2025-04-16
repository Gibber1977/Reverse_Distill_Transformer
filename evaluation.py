# --- Evaluation Module ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(Y_true_test, Y_pred_test, scaler, target_col_index, model_name="Model"):
    print(f"\nEvaluating {model_name} predictions...")

    # Inverse transform ground truth Y_true_test (which is already scaled)
    num_samples = Y_true_test.shape[0]
    num_features = scaler.n_features_in_
    pred_len = Y_true_test.shape[1]

    Y_true_test_inv = np.zeros((num_samples, pred_len, 1))

    for step in range(pred_len):
        dummy_array = np.zeros((num_samples, num_features))
        dummy_array[:, target_col_index] = Y_true_test[:, step, 0]
        inversed_array = scaler.inverse_transform(dummy_array)
        Y_true_test_inv[:, step, 0] = inversed_array[:, target_col_index]

    print(f"Shape of inverse-scaled True Y: {Y_true_test_inv.shape}")
    print(f"Shape of inverse-scaled Predicted Y: {Y_pred_test.shape}")

    # Calculate metrics for each prediction step
    mses, maes = [], []
    for step in range(pred_len):
        mse = mean_squared_error(Y_true_test_inv[:, step, 0], Y_pred_test[:, step, 0])
        mae = mean_absolute_error(Y_true_test_inv[:, step, 0], Y_pred_test[:, step, 0])
        mses.append(mse)
        maes.append(mae)
        print(f"Step {step+1}/{pred_len} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    avg_mse = np.mean(mses)
    avg_mae = np.mean(maes)
    print(f"\n{model_name} Average over {pred_len} steps - MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")

    # --- Visualization ---
    # Plot predictions vs actual for a sample period from the test set
    plt.figure(figsize=(15, 6))
    # Plotting the first prediction step only for clarity
    plot_len = min(200, len(Y_true_test_inv)) # Plot first 200 points or less
    plt.plot(Y_true_test_inv[:plot_len, 0, 0], label='Actual Temperature (Step 1)', color='blue')
    plt.plot(Y_pred_test[:plot_len, 0, 0], label='Predicted Temperature (Step 1)', color='red', linestyle='--')
    plt.title(f'{model_name} Temperature Prediction vs Actual (Test Set - First Step)')
    plt.xlabel('Time Steps (in test set)')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = f"{model_name.lower().replace(' ', '_')}_prediction_vs_actual.png"
    plt.savefig(plot_filename)
    print(f"Saved prediction plot to {plot_filename}")
    plt.close()
    
    return avg_mse, avg_mae