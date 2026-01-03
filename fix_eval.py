import tensorflow as tf
import os
from config import BATCH_SIZE
from data_loader import get_dataset
from model import build_model
from utils import calculate_average_metrics

def main():
    print("=== Starting Fixed Evaluation ===")
    
    # 1. Load Data
    print("[1/3] Loading Test Data...")
    test_dataset = get_dataset(batch_size=BATCH_SIZE, augment=False)
    
    # 2. Build Model & Load Weights
    print("[2/3] Building Model and Loading Weights...")
    model = build_model()
    
    weights_path = 'model.weights.h5'
    if os.path.exists(weights_path):
        print(f"Found weights at {weights_path}")
        try:
            model.load_weights(weights_path)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Error: {weights_path} not found.")
        return

    # 3. Calculate Metrics
    print("[3/3] Calculating Metrics...")
    psnr, ssim, mse, mae = calculate_average_metrics(model, test_dataset, num_batches=20)
    
    print("\n" + "="*30)
    print(f"RESULTS:")
    print(f"Average PSNR: {psnr:.4f}")
    print(f"Average SSIM: {ssim:.4f}")
    print(f"Average MSE:  {mse:.6f}")
    print(f"Average MAE:  {mae:.6f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
