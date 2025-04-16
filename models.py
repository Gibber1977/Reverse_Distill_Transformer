# --- Models Module ---
import torch
import torch.nn as nn
import math
from sklearn.linear_model import LinearRegression
import numpy as np
from config import SEQ_LEN

# --- Linear Teacher Model ---
def train_and_predict_linear_teacher(X_train, Y_true_train, X_val, pred_len, n_features):
    print("Training Linear Teacher Model...")
    # Reshape X for scikit-learn LinearRegression (samples, features*seq_len)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # Reshape Y (samples, pred_len) - assuming single target feature prediction
    Y_true_train_flat = Y_true_train.reshape(Y_true_train.shape[0], -1)

    teacher_model = LinearRegression()
    teacher_model.fit(X_train_flat, Y_true_train_flat)

    print("Generating Teacher Predictions (Soft Labels)...")
    Y_linear_train_flat = teacher_model.predict(X_train_flat)
    Y_linear_val_flat = teacher_model.predict(X_val_flat)

    # Reshape back to (samples, pred_len, 1) to match student output
    Y_linear_train = Y_linear_train_flat.reshape(-1, pred_len, 1)
    Y_linear_val = Y_linear_val_flat.reshape(-1, pred_len, 1)

    print(f"Teacher prediction shapes: Train={Y_linear_train.shape}, Val={Y_linear_val.shape}")
    return teacher_model, Y_linear_train, Y_linear_val

# --- Transformer Student Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerStudentModel(nn.Module):
    def __init__(self, n_features, d_model, n_heads, num_encoder_layers, dim_feedforward, pred_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        # Output layer projects the encoding of the *last* sequence element (or average?)
        # Let's try projecting the output of all sequence elements and then averaging/flattening
        self.output_projection1 = nn.Linear(d_model * SEQ_LEN, d_model * 2) # Project flattened encoder output
        self.relu = nn.ReLU()
        self.output_projection2 = nn.Linear(d_model * 2, pred_len * 1) # Predict pred_len steps for 1 target feature
        self.pred_len = pred_len

    def forward(self, src):
        # src shape: (batch_size, seq_len, n_features)
        src = self.input_projection(src) * math.sqrt(self.d_model)
        # Transformer Encoder expects (batch_size, seq_len, d_model) with batch_first=True
        output = self.transformer_encoder(src) # shape: (batch_size, seq_len, d_model)

        # Flatten the output sequence encoding for the final linear layers
        output = output.reshape(output.size(0), -1) # (batch_size, seq_len * d_model)

        output = self.relu(self.output_projection1(output))
        output = self.output_projection2(output) # (batch_size, pred_len * 1)

        # Reshape to (batch_size, pred_len, n_target_features=1)
        output = output.view(-1, self.pred_len, 1)
        return output