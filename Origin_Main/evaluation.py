# --- Evaluation Module ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero, handle cases where y_true is close to zero
    # You might want to adjust the tolerance based on your data
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def evaluate_predictions(Y_true_test, Y_pred_test, scaler, target_col_index, model_name="Model"):
    print(f"\nEvaluating {model_name} predictions...")

    # Ensure Y_true_test and Y_pred_test have the same shape for metrics calculation
    # Although inverse transform might change the last dimension, we need to align them for metrics
    # Assuming Y_true_test[:, :, 0] and Y_pred_test[:, :, 0] represent the predictions
    if Y_true_test.shape[-1] > 1:
         Y_true_test_processed = Y_true_test[:, :, 0]
    else:
         Y_true_test_processed = Y_true_test[:, :, 0] # Keep the last dim if it's 1

    if Y_pred_test.shape[-1] > 1:
        Y_pred_test_processed = Y_pred_test[:, :, 0]
    else:
        Y_pred_test_processed = Y_pred_test[:, :, 0] # Keep the last dim if it's 1


    # Inverse transform ground truth Y_true_test (which is already scaled)
    num_samples = Y_true_test_processed.shape[0]
    pred_len = Y_true_test_processed.shape[1]

    # Prepare storage for inverse transformed data
    Y_true_test_inv = np.zeros((num_samples, pred_len))

    # Need the total number of features the scaler was fitted on to inverse transform correctly
    # Assuming the scaler was fitted on the original training data shape
    if not hasattr(scaler, 'n_features_in_') or scaler.n_features_in_ is None:
        print("Warning: scaler.n_features_in_ is not available. Assuming original feature count for inverse transform.")
        # This requires knowing the original number of features. You might need to pass it as an argument.
        # For now, as a fallback, let's assume it's the number of features in the data used to train the model.
        # This is a potential point of failure if the context isn't clear.
        # A better approach is to pass the original number of features or the shape of the data used to fit the scaler.
        # For demonstration, let's assume a default or try to infer (risky).
        # In a real application, pass the original number of features or the data shape.
        # Assuming the data fed to the scaler had shape (samples, original_features)
        # This is hard to infer correctly without more context.
        # Let's use a placeholder and highly recommend passing the correct original feature count.
        original_num_features = 10 # !!! Placeholder - Replace with the actual number of features the scaler was trained on !!!
        print(f"Using a placeholder original_num_features: {original_num_features}. Please ensure this is correct.")
    else:
        original_num_features = scaler.n_features_in_


    for step in range(pred_len):
        # Create a dummy array with the same number of features the scaler expects
        dummy_array = np.zeros((num_samples, original_num_features))
        # Place the scaled target column data into the correct position in the dummy array
        dummy_array[:, target_col_index] = Y_true_test_processed[:, step]

        try:
            inversed_array = scaler.inverse_transform(dummy_array)
            Y_true_test_inv[:, step] = inversed_array[:, target_col_index]
        except Exception as e:
            print(f"Error during inverse transform at step {step}: {e}")
            print("Please check if the scaler was fitted correctly and if target_col_index is valid.")
            # If inverse transform fails, metrics will be misleading. We should probably exit or handle this.
            # For now, let's print the error and continue, but be aware the results might be incorrect.
            # You might want to add more robust error handling.


    # Ensure Y_pred_test also has the correct shape for metrics (num_samples, pred_len)
    # Assuming Y_pred_test shape is (num_samples, pred_len) or (num_samples, pred_len, 1)
    if Y_pred_test.ndim == 3:
        Y_pred_test_processed = Y_pred_test[:, :, 0]
    elif Y_pred_test.ndim == 2:
         Y_pred_test_processed = Y_pred_test # Already in (num_samples, pred_len) format
    else:
         raise ValueError(f"Unexpected shape for Y_pred_test: {Y_pred_test.shape}. Expected 2 or 3 dimensions.")


    # Need to inverse transform Y_pred_test as well if it's scaled
    # If Y_pred_test comes directly from the model and the model outputs scaled values, it needs inverse scaling.
    # If the model outputs inverse scaled values, then Y_pred_test_processed should be used directly.
    # Assuming Y_pred_test is scaled and needs inverse transformation similar to Y_true_test
    Y_pred_test_inv = np.zeros((num_samples, pred_len))
    for step in range(pred_len):
         dummy_array_pred = np.zeros((num_samples, original_num_features))
         dummy_array_pred[:, target_col_index] = Y_pred_test_processed[:, step]
         try:
             inversed_array_pred = scaler.inverse_transform(dummy_array_pred)
             Y_pred_test_inv[:, step] = inversed_array_pred[:, target_col_index]
         except Exception as e:
             print(f"Error during inverse transform of predictions at step {step}: {e}")
             # Handle this error similar to the true value inverse transform failure


    print(f"Shape of inverse-scaled True Y: {Y_true_test_inv.shape}")
    print(f"Shape of inverse-scaled Predicted Y: {Y_pred_test_inv.shape}")

    # Calculate metrics for each prediction step
    mses, rmses, maes, mapes, r2s = [], [], [], [], []
    print("\nMetrics per prediction step:")
    for step in range(pred_len):
        true_step = Y_true_test_inv[:, step]
        pred_step = Y_pred_test_inv[:, step]

        # Handle potential NaNs or infs after inverse transform
        valid_indices = np.isfinite(true_step) & np.isfinite(pred_step)
        true_step_valid = true_step[valid_indices]
        pred_step_valid = pred_step[valid_indices]

        if len(true_step_valid) == 0:
            print(f"Step {step+1}/{pred_len}: No valid data points for evaluation.")
            mses.append(np.nan)
            rmses.append(np.nan)
            maes.append(np.nan)
            mapes.append(np.nan)
            r2s.append(np.nan)
            continue

        mse = mean_squared_error(true_step_valid, pred_step_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_step_valid, pred_step_valid)
        # Handle potential issues with MAPE if true_step_valid contains zeros
        mape = mean_absolute_percentage_error(true_step_valid, pred_step_valid)
        # R2 score can be negative, indicating poor fit
        r2 = r2_score(true_step_valid, pred_step_valid)

        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)
        r2s.append(r2)

        print(f"Step {step+1}/{pred_len} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")

    # Calculate average metrics, ignoring NaNs
    avg_mse = np.nanmean(mses)
    avg_rmse = np.nanmean(rmses)
    avg_mae = np.nanmean(maes)
    avg_mape = np.nanmean(mapes)
    avg_r2 = np.nanmean(r2s)

    print(f"\n--- {model_name} Average over {pred_len} steps (ignoring NaNs) ---")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average R2 Score: {avg_r2:.4f}")
    print("-------------------------------------------------------")


    # --- Visualization ---
    results_folder = 'results' # Create a folder for results if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    model_folder = os.path.join(results_folder, model_name.lower().replace(' ', '_'))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


    # Plot predictions vs actual for a sample period from the test set (First prediction step)
    plt.figure(figsize=(15, 6))
    plot_len = min(200, len(Y_true_test_inv)) # Plot first 200 points or less
    plt.plot(Y_true_test_inv[:plot_len, 0], label='Actual (Step 1)', color='blue')
    plt.plot(Y_pred_test_inv[:plot_len, 0], label='Predicted (Step 1)', color='red', linestyle='--')
    plt.title(f'{model_name} Prediction vs Actual (Test Set - First Step)')
    plt.xlabel('Time Steps (in test set)')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plot_filename = f"{model_name.lower().replace(' ', '_')}_prediction_vs_actual_step1.png"
    plot_path = os.path.join(model_folder, plot_filename)
    plt.savefig(plot_path)
    print(f"Saved prediction vs actual plot (Step 1) to {plot_path}")
    plt.close()

    # Plot actual vs predicted scatter plot for the first prediction step
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_true_test_inv[:, 0], Y_pred_test_inv[:, 0], alpha=0.5)
    max_val = max(Y_true_test_inv[:, 0].max(), Y_pred_test_inv[:, 0].max())
    min_val = min(Y_true_test_inv[:, 0].min(), Y_pred_test_inv[:, 0].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--') # Diagonal line for perfect prediction
    plt.title(f'{model_name} Actual vs Predicted (Test Set - Step 1)')
    plt.xlabel('Actual Value (Step 1)')
    plt.ylabel('Predicted Value (Step 1)')
    plt.grid(True)
    scatter_filename = f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted_scatter_step1.png"
    scatter_path = os.path.join(model_folder, scatter_filename)
    plt.savefig(scatter_path)
    print(f"Saved actual vs predicted scatter plot (Step 1) to {scatter_path}")
    plt.close()


    # Plot MSE, RMSE, MAE, R2 over prediction steps
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, pred_len + 1), mses)
    plt.title(f'{model_name} MSE per Step')
    plt.xlabel('Prediction Step')
    plt.ylabel('MSE')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, pred_len + 1), rmses)
    plt.title(f'{model_name} RMSE per Step')
    plt.xlabel('Prediction Step')
    plt.ylabel('RMSE')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, pred_len + 1), maes)
    plt.title(f'{model_name} MAE per Step')
    plt.xlabel('Prediction Step')
    plt.ylabel('MAE')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, pred_len + 1), r2s)
    plt.title(f'{model_name} R2 Score per Step')
    plt.xlabel('Prediction Step')
    plt.ylabel('R2 Score')
    plt.grid(True)

    plt.tight_layout()
    metrics_plot_filename = f"{model_name.lower().replace(' ', '_')}_metrics_per_step.png"
    metrics_plot_path = os.path.join(model_folder, metrics_plot_filename)
    plt.savefig(metrics_plot_path)
    print(f"Saved metrics per step plot to {metrics_plot_path}")
    plt.close()


    # Plot MAPE per step
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, pred_len + 1), mapes)
    plt.title(f'{model_name} MAPE per Step')
    plt.xlabel('Prediction Step')
    plt.ylabel('MAPE (%)')
    plt.grid(True)
    mape_plot_filename = f"{model_name.lower().replace(' ', '_')}_mape_per_step.png"
    mape_plot_path = os.path.join(model_folder, mape_plot_filename)
    plt.savefig(mape_plot_path)
    print(f"Saved MAPE per step plot to {mape_plot_path}")
    plt.close()


    return avg_mse, avg_rmse, avg_mae, avg_mape, avg_r2

# Example Usage (assuming you have Y_true_test, Y_pred_test, scaler, and target_col_index defined)
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# # Dummy data for demonstration
# scaler = MinMaxScaler()
# # Assuming scaler was fitted on data with 5 features
# original_data_shape = (100, 5) # Placeholder
# scaler.fit(np.random.rand(*original_data_shape))

# # Dummy test data - scaled
# # Y_true_test shape: (samples, prediction_length, 1)
# Y_true_test_scaled = np.random.rand(50, 10, 1)
# # Y_pred_test shape: (samples, prediction_length) or (samples, prediction_length, 1)
# Y_pred_test_scaled = np.random.rand(50, 10) # Example where model outputs (samples, prediction_length)


# # Specify the index of the target column in the original feature space
# target_col_index = 2 # Example: Target was the 3rd feature in the original data

# # If your model outputs already inverse-scaled values, pass them directly
# # For this example, assuming Y_pred_test_scaled needs inverse scaling
# # Let's simulate the inverse scaling for Y_pred_test_scaled here for the example
# Y_pred_test_simulated_inv = np.zeros_like(Y_pred_test_scaled)
# dummy_array_pred_example = np.zeros((Y_pred_test_scaled.shape[0], scaler.n_features_in_))
# for step in range(Y_pred_test_scaled.shape[1]):
#      dummy_array_pred_example[:, target_col_index] = Y_pred_test_scaled[:, step]
#      inversed_array_pred_example = scaler.inverse_transform(dummy_array_pred_example)
#      Y_pred_test_simulated_inv[:, step] = inversed_array_pred_example[:, target_col_index]


# # Now call the evaluation function
# # Pass the inverse-scaled predictions if your model outputs them directly,
# # otherwise pass the scaled predictions and the function will inverse scale them.
# # In this modified function, we assume Y_pred_test is scaled and needs inverse transform.
# # So, pass Y_pred_test_scaled as the second argument.
# avg_mse, avg_rmse, avg_mae, avg_mape, avg_r2 = evaluate_predictions(
#     Y_true_test_scaled,
#     Y_pred_test_scaled,
#     scaler,
#     target_col_index,
#     model_name="MyTimeSeriesModel"
# )

# print("\n--- Overall Evaluation Summary ---")
# print(f"Model: MyTimeSeriesModel")
# print(f"Average MSE: {avg_mse:.4f}")
# print(f"Average RMSE: {avg_rmse:.4f}")
# print(f"Average MAE: {avg_mae:.4f}")
# print(f"Average MAPE: {avg_mape:.2f}%")
# print(f"Average R2 Score: {avg_r2:.4f}")
# print("-----------------------------------")

