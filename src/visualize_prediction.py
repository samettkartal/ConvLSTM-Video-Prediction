import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imageio
from .config import BATCH_SIZE, RESULTS_DIR
from .model import build_model
from .data_loader import create_shifted_frames
from .utils import load_trained_model

def create_gif_visual(model, dataset, save_path):
    """
    Creates a high-quality GIF visualization for the presentation.
    Side-by-side comparison: 
    Left: Ground Truth
    Right: Model Prediction
    """
    print("Generating GIF visualization...")
    
    # Take one batch
    for x, y in dataset.take(1):
        # Predict
        preds = model.predict(x, verbose=0)
        
        # Select the first sample in the batch
        i = 0 
        
        frames = []
        
        # Iterate through the prediction length (Time steps)
        # y shape: (B, T, H, W, C). T=19 usually
        num_frames = y.shape[1]
        
        for t in range(num_frames):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Left: Ground Truth
            # Reverse color map for better visibility if needed, but gray is standard
            axes[0].imshow(y[i, t, :, :, 0], cmap='gray') 
            axes[0].set_title(f"Beklenen (Gerçek) - t={t+1}", fontsize=14)
            axes[0].axis('off')
            
            # Right: Prediction
            axes[1].imshow(preds[i, t, :, :, 0], cmap='gray')
            axes[1].set_title(f"Model Tahmini - t={t+1}", fontsize=14)
            axes[1].axis('off')
            
            # Save frame to buffer
            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            # method 1: buffer_rgba (most compatible with recent matplotlib)
            try:
                data = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                w, h = fig.canvas.get_width_height()
                image = data.reshape((h, w, 4))
                # Discard alpha channel to get RGB
                image = image[:, :, :3] 
            except AttributeError:
                # Fallback for very old versions or different backends, though unlikely in Colab
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                w, h = fig.canvas.get_width_height()
                image = data.reshape((h, w, 3))
            
            frames.append(image)
            plt.close()
            
        # Save GIF
        imageio.mimsave(save_path, frames, fps=2) # 2 frames per second
        print(f"GIF visualization saved to {save_path}")
        return # Only do one sample

def main():
    # 1. Load Model
    model = load_trained_model()
    if model is None:
        return

    # 2. Load Data
    dataset = get_dataset(batch_size=BATCH_SIZE, augment=False)

    # 3. Generate Visual
    save_path = os.path.join(RESULTS_DIR, 'sunum_karsilastirma.gif')
    create_gif_visual(model, dataset, save_path)

if __name__ == "__main__":
    main()
