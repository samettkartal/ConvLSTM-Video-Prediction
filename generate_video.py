import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from config import RESULTS_DIR
from data_loader import get_dataset
from utils import load_trained_model

def create_animation(sample_idx, inputs, video_preds, save_path):
    """
    Creates a GIF animation:
    1. Shows Input frames (Real context)
    2. Shows Predicted frames
    """
    # Inputs: (T_in, H, W, C)
    # Preds: (T_pred, H, W, C)
    
    # Combine sequence: Inputs -> Preds
    # Note: inputs and video_preds are single sample arrays (T, H, W, C)
    
    full_sequence = np.concatenate([inputs, video_preds], axis=0)
    # full_sequence shape: (Total_Frames, H, W, C)
    
    total_frames = full_sequence.shape[0]
    input_len = inputs.shape[0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.axis('off')
    
    # Initial frame
    img_plot = ax.imshow(full_sequence[0, :, :, 0], cmap='gray')
    title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=14)
    
    def update(frame_idx):
        img_plot.set_data(full_sequence[frame_idx, :, :, 0])
        
        if frame_idx < input_len:
            label = f"Input (Real) - Frame {frame_idx + 1}/{input_len}"
            color = 'black'
        else:
            pred_idx = frame_idx - input_len
            label = f"Prediction - Frame {pred_idx + 1}/{total_frames - input_len}"
            color = 'blue'
            
        title_text.set_text(label)
        title_text.set_color(color)
        return [img_plot, title_text]

    # Interval is in milliseconds. 1000ms = 1 second.
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        interval=1000, 
        blit=True
    )
    
    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer='pillow', fps=1)
    plt.close()

def main():
    print("=== Generating Video/GIF Outputs ===")
    
    # 1. Load Model
    model = load_trained_model()
    if model is None:
        print("Failed to load model.")
        return

    # 2. Get Data (No shuffle to be consistent, batch size 1 for easy iterating)
    dataset = get_dataset(batch_size=1, augment=False)
    
    # 3. Generate for a few samples
    num_samples_to_generate = 3
    
    print(f"Generating videos for {num_samples_to_generate} samples...")
    
    for i, (x_batch, y_batch) in enumerate(dataset.take(num_samples_to_generate)):
        # x_batch: (1, 10, 64, 64, 1)
        # y_batch: (1, 10, 64, 64, 1) - Targets (Ground Truth)
        
        preds = model.predict(x_batch, verbose=0)
        # preds: (1, 10, 64, 64, 1)
        
        # Remove batch dimension
        x_sample = x_batch[0]   # (10, 64, 64, 1)
        pred_sample = preds[0]  # (10, 64, 64, 1)
        
        # Ensure directory exists
        video_dir = os.path.join(RESULTS_DIR, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        save_file = os.path.join(video_dir, f'prediction_sample_{i+1}.gif')
        
        create_animation(i, x_sample, pred_sample, save_file)
        
    print(f"Done! Videos saved in {video_dir}")

if __name__ == "__main__":
    main()
