# --- Training Module ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def get_alpha(current_epoch, total_epochs, alpha_start, alpha_end):
    """Linearly increase alpha from start to end over epochs."""
    return alpha_start + (alpha_end - alpha_start) * (current_epoch / total_epochs)

def train_rdt_model(student_model, X_train, Y_true_train, Y_linear_train, X_val, Y_true_val, Y_linear_val,
                    epochs, batch_size, lr, alpha_start, alpha_end, device):
    print("Starting RDT Training...")
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device),
                                  torch.FloatTensor(Y_true_train).to(device),
                                  torch.FloatTensor(Y_linear_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(X_val).to(device),
                                torch.FloatTensor(Y_true_val).to(device),
                                torch.FloatTensor(Y_linear_val).to(device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(student_model.parameters(), lr=lr)
    criterion_task = nn.MSELoss()
    criterion_distill = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        student_model.train()
        total_train_loss = 0
        total_task_loss_train = 0
        total_distill_loss_train = 0

        current_alpha = get_alpha(epoch, epochs, alpha_start, alpha_end)
        print(f"\nEpoch {epoch+1}/{epochs}, Current Alpha: {current_alpha:.4f}")

        for X_batch, Y_true_batch, Y_linear_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            Y_pred_batch = student_model(X_batch)

            # Calculate losses
            loss_task = criterion_task(Y_pred_batch, Y_true_batch)
            loss_distill = criterion_distill(Y_pred_batch, Y_linear_batch)

            # Combine losses
            loss_total = current_alpha * loss_task + (1 - current_alpha) * loss_distill

            # Backward pass and optimize
            loss_total.backward()
            optimizer.step()

            total_train_loss += loss_total.item()
            total_task_loss_train += loss_task.item()
            total_distill_loss_train += loss_distill.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_task_loss_train = total_task_loss_train / len(train_loader)
        avg_distill_loss_train = total_distill_loss_train / len(train_loader)
        print(f"Avg Train Loss: {avg_train_loss:.4f} (Task: {avg_task_loss_train:.4f}, Distill: {avg_distill_loss_train:.4f})")

        # Validation step
        student_model.eval()
        total_val_loss_task = 0
        with torch.no_grad():
            for X_batch_val, Y_true_batch_val, Y_linear_batch_val in val_loader:
                Y_pred_batch_val = student_model(X_batch_val)
                val_loss_task = criterion_task(Y_pred_batch_val, Y_true_batch_val)
                # We primarily care about task loss on validation for model selection
                total_val_loss_task += val_loss_task.item()

        avg_val_loss_task = total_val_loss_task / len(val_loader)
        print(f"Avg Validation Task Loss (MSE): {avg_val_loss_task:.4f}")

        # Save best model based on validation task loss
        if avg_val_loss_task < best_val_loss:
            best_val_loss = avg_val_loss_task
            best_model_state = student_model.state_dict()
            print(f"*** New best validation loss: {best_val_loss:.4f}. Saving model state. ***")

    # Load best model weights
    if best_model_state:
        student_model.load_state_dict(best_model_state)
        print("Loaded best model weights based on validation loss.")
    else:
        print("Warning: No best model state saved (validation loss might not have improved).")

    return student_model