# --- Configuration ---
import torch

# Data configuration
DATA_PATH = 'weatherHistory.csv'
TARGET_COLUMN = 'Temperature (C)'
# Select features relevant for prediction + the target itself
FEATURE_COLUMNS = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
SEQ_LEN = 24  # Use past 24 hours
PRED_LEN = 3   # Predict next 3 hours

# Training configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10 # Keep low for quick testing
ALPHA_START = 0.2 # Initial weight for task loss
ALPHA_END = 0.8   # Final weight for task loss (approached linearly over epochs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
D_MODEL = 64
N_HEADS = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1