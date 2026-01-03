import tensorflow as tf
from config import BATCH_SIZE
from data_loader import get_dataset
from model import build_model
import os

def test_pipeline():
    print("Testing Data Pipeline and Model Training...")
    
    # 1. Load Data (Small subset)
    try:
        dataset = get_dataset(batch_size=2, augment=False)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 2. Build Model
    try:
        model = build_model()
        model.summary()
        print("Model built successfully.")
    except Exception as e:
        print(f"Model building failed: {e}")
        return

    # 3. Train for 1 step
    print("Attempting to train for 1 step...")
    try:
        model.fit(dataset.take(1), epochs=1, verbose=1)
        print("Training step successful.")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
