
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import RESULTS_DIR, BATCH_SIZE
from data_loader import get_dataset
from generate_presentation_assets import robust_load_model

def load_benchmark_model(path):
    print(f"Loading benchmark model from {path}...")
    try:
        # Try generic load
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"Failed to load benchmark: {e}")
        return None

def create_triple_comparison_gif(sample_idx, x_input, gt, pred_ours, pred_bench, save_path):
    """
    Left: GT
    Middle: Benchmark
    Right: Ours
    """
    # x_input: (19, 64, 64, 1) -> T=0..18
    # GT, Preds: (19, 64, 64, 1) -> T=1..19
    
    # Construct sequences
    # For display, we usually show the PREDICTED sequence T=1..19
    # But strictly, T=0 is known.
    # Let's show T=1..19 (Targets)
    
    # Clip to 0-1
    pred_ours = np.clip(pred_ours, 0, 1)
    if pred_bench is not None:
        pred_bench = np.clip(pred_bench, 0, 1)
    
    num_frames = gt.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(top=0.85)

    # Initialize
    # Left: GT
    im_gt = axes[0].imshow(gt[0, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    
    # Middle: Benchmark
    if pred_bench is not None:
        im_ben = axes[1].imshow(pred_bench[0, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title("Benchmark Model", fontsize=14)
    else:
        im_ben = axes[1].imshow(np.zeros((64,64)), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title("Benchmark (Missing)", fontsize=14)
        print("Benchmark predictions missing for GIF.")
    axes[1].axis('off')
    
    # Right: Ours
    im_our = axes[2].imshow(pred_ours[0, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[2].set_title("Our Model (Ours)", fontsize=14, color='blue')
    axes[2].axis('off')
    
    title = plt.suptitle(f"Frame 1", fontsize=16)

    def update(i):
        im_gt.set_data(gt[i, :, :, 0])
        if pred_bench is not None:
            im_ben.set_data(pred_bench[i, :, :, 0])
        im_our.set_data(pred_ours[i, :, :, 0])
        title.set_text(f"Frame {i+1}")
        return [im_gt, im_ben, im_our, title]
        
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)
    ani.save(save_path, writer='pillow', fps=5, savefig_kwargs={'dpi': 150})
    plt.close()
    print(f"Comparison GIF saved to {save_path}")

def main():
    print("=== Model Comparison Benchmark ===")
    
    # 1. Load Our Model
    print("Loading our model...")
    our_model = robust_load_model() # This loads from 'results' or 'checkpoints'
    if our_model is None:
        print("Could not load our model! Run training first.")
        return

    # 2. Load Benchmark Model
    # Expecting explicit file
    bench_path = os.path.join(RESULTS_DIR, "benchmark_model.weights.h5")
    # Also check .keras or .h5
    if not os.path.exists(bench_path):
        bench_path = os.path.join(RESULTS_DIR, "benchmark_model.keras")
    if not os.path.exists(bench_path):
        bench_path = os.path.join(RESULTS_DIR, "benchmark_model.h5") # Plain H5
        
    bench_model = None
    if os.path.exists(bench_path):
        bench_model = load_benchmark_model(bench_path)
    else:
        print(f"Benchmark model not found in {RESULTS_DIR}. Comparison will be partial.")
        
    # 3. Load Data
    dataset = get_dataset(batch_size=8, augment=False)
    
    # 4. Generate Predictions & Visuals
    save_dir = os.path.join(RESULTS_DIR, "presentation_assets")
    os.makedirs(save_dir, exist_ok=True)
    
    for x_batch, y_batch in dataset.take(1):
        print("Generating predictions...")
        preds_ours = our_model.predict(x_batch, verbose=0)
        
        preds_bench = None
        if bench_model:
            try:
                preds_bench = bench_model.predict(x_batch, verbose=0)
            except Exception as e:
                print(f"Benchmark inference failed: {e}")
        
        # Create 1 GIF
        print("Creating GIF...")
        # Index 2 often has good motion
        idx = 2 if x_batch.shape[0] > 2 else 0
        
        save_path = os.path.join(save_dir, "benchmark_comparison.gif")
        create_triple_comparison_gif(idx, x_batch[idx].numpy(), y_batch[idx].numpy(), 
                                     preds_ours[idx], preds_bench[idx] if preds_bench is not None else None, 
                                     save_path)
                                     
        # Save a static frame comparison too (Middle of sequence)
        frame_idx = 10
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(y_batch[idx, frame_idx, :, :, 0], cmap='gray', interpolation='nearest')
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        
        if preds_bench is not None:
            axes[1].imshow(preds_bench[idx, frame_idx, :, :, 0], cmap='gray', interpolation='nearest')
            axes[1].set_title("Benchmark")
        else:
            axes[1].imshow(np.zeros((64,64)), cmap='gray', interpolation='nearest')
            axes[1].text(0.5, 0.5, "Missing", ha='center', color='white')
            axes[1].set_title("Benchmark (Missing)")
        axes[1].axis('off')

        axes[2].imshow(preds_ours[idx, frame_idx, :, :, 0], cmap='gray', interpolation='nearest')
        axes[2].set_title("Ours")
        axes[2].axis('off')
        
        plt.suptitle(f"Comparison at Frame {frame_idx}")
        plt.savefig(os.path.join(save_dir, "benchmark_static.png"), dpi=300)
        print("Static comparison saved.")
        break

if __name__ == "__main__":
    main()
