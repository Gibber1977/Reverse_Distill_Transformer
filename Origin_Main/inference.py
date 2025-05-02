# --- Inference Module ---
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config import PRED_LEN

def predict_on_test(model, X_test, scaler, target_col_index, batch_size, device, is_linear=False):
    print("Running inference on test set...")
    
    if is_linear:
        # Handle linear model case
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        Y_pred_test_scaled = model.predict(X_test_flat)
        Y_pred_test_scaled = Y_pred_test_scaled.reshape(-1, PRED_LEN, 1)
    else:
        # Handle neural network case
        model.eval()
        test_dataset = TensorDataset(torch.FloatTensor(X_test).to(device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_preds_scaled = []
        with torch.no_grad():
            for X_batch, in test_loader: # Only need X for prediction
                Y_pred_batch_scaled = model(X_batch)
                all_preds_scaled.append(Y_pred_batch_scaled.cpu().numpy())

        # Concatenate predictions from all batches
        Y_pred_test_scaled = np.concatenate(all_preds_scaled, axis=0)
    
    print(f"Raw prediction shape (scaled): {Y_pred_test_scaled.shape}") # Should be (n_test_samples, pred_len, 1)

    # Inverse transform predictions
    # Scaler expects (n_samples, n_features). We need to carefully place our predicted target values
    # into an array of the correct shape, inverse transform, and then extract the target column.

    # Create a dummy array with the shape scaler expects for the number of predictions made
    # We need to repeat the prediction for each time step to fill the feature dimension
    num_samples = Y_pred_test_scaled.shape[0]
    num_features = scaler.n_features_in_
    pred_len = Y_pred_test_scaled.shape[1]

    # Placeholder for inverse transformation - needs careful handling per timestep
    Y_pred_test_inv = np.zeros((num_samples, pred_len, 1))

    # Process each prediction step separately for inverse scaling
    for step in range(pred_len):
        dummy_array = np.zeros((num_samples, num_features))
        # Place the scaled predictions for this step into the target column
        dummy_array[:, target_col_index] = Y_pred_test_scaled[:, step, 0]
        # Inverse transform
        inversed_array = scaler.inverse_transform(dummy_array)
        # Extract the inverse-transformed target column for this step
        Y_pred_test_inv[:, step, 0] = inversed_array[:, target_col_index]

    print(f"Final prediction shape (inverse-scaled): {Y_pred_test_inv.shape}")
    return Y_pred_test_inv