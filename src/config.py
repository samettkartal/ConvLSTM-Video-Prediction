import os

# Hyperparameters
# Hyperparameters
BATCH_SIZE = 8      # Reduced from 32 (Reference uses 5)
EPOCHS = 50
LR = 5e-4           # Slightly reduced for smaller batches
IMG_SIZE = (64, 64)
SEQ_LEN = 20  # Total sequence length
INPUT_LEN = 19  # Input: Frames 0-18
PRED_LEN = 19   # Target: Frames 1-19
CHANNELS = 1

# Model Architecture Params
# Model Architecture Params
FILTERS = 64
# Kernel sizes for layers 1, 2, 3 respectively
KERNEL_SIZES = [(5, 5), (3, 3), (1, 1)] 
USE_LAYER_NORM = True # Kept for legacy compatibility
DROPOUT_RATE = 0.25 # Kept for optional use, though reference doesn't emphasize it much
L2_REG = 1e-4

# Loss Weights
# BCE is the primary loss for binary pixels (Scale ~0.02 - 0.05)
BCE_WEIGHT = 10.0   # Increased to 10.0 to FORCE content correctness
PERCEPTUAL_WEIGHT = 0.1     
GDL_WEIGHT = 0.1    # Reduced to 0.1 to prevent "Halo" artifacts (black edges)
GDL_ALPHA = 1.0             # L1-like norm for sharper edges

# Training Stability
GRADIENT_CLIP_NORM = 1.0   # Clip gradients to prevent explosion

# Paths
# Paths
try:
    # Go up one level from 'src' to get to the project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
