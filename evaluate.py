import os
from src.config import *
from src.data_loader import create_shifted_frames
from src.model import build_model
from src.utils import save_images

def main():
    print("=== Starting Evaluation ===")
    
    # 1. Load Data
    print("[1/4] Loading Test Data...")
    test_dataset = get_dataset(batch_size=BATCH_SIZE, augment=False)
    
    # 2. Load Model
    print("[2/4] Loading Model...")
    model = load_trained_model()
    if model is None:
        print("Error: Could not load model.")
        return

    # 3. Calculate Metrics
    # 3. Calculate Metrics
    print("[3/4] Calculating Metrics (PSNR, SSIM, MSE, MAE)...")
    psnr, ssim, mse, mae = calculate_average_metrics(model, test_dataset, num_batches=20)
    
    print("\n" + "="*30)
    print(f"RESULTS:")
    print(f"Average PSNR: {psnr:.4f}")
    print(f"Average SSIM: {ssim:.4f}")
    print(f"Average MSE:  {mse:.6f}")
    print(f"Average MAE:  {mae:.6f}")
    print("="*30 + "\n")

    # 4. Visualization
    print("[4/4] Generating Visualizations...")
    save_path = os.path.join(RESULTS_DIR, 'evaluation_results.png')
    visualize_predictions(model, test_dataset, save_path, num_samples=5)
    
    print(f"Evaluation Complete! Check {save_path} for visualizations.")

if __name__ == "__main__":
    main()
