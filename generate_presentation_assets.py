

import os
import sys
import zipfile
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import RESULTS_DIR, BATCH_SIZE, CHECKPOINT_DIR
from data_loader import get_dataset
from utils import calculate_average_metrics
# Import build_model explicitly for the fallback strategy
from model import build_model 

def robust_load_model(checkpoint_dir=CHECKPOINT_DIR):
    """
    Attempts to load model using multiple strategies to handle TF version mismatches.
    Strategy 1: Standard tf.keras.models.load_model (Success if versions match)
    Strategy 2: Extract weights from .keras (Zip) and load into build_model() (Backwards compatibility)
    """
    model_path = os.path.join(checkpoint_dir, 'model_best.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, 'model_final.keras')
    
    if not os.path.exists(model_path):
        print(f"No model found in {checkpoint_dir}")
        return None

    print(f"Attempting to load model from {model_path}...")

    # Strategy 1: Direct Load
    try:
        from utils import load_trained_model as original_load
        model = original_load(checkpoint_dir)
        if model is not None:
             return model
    except Exception as e:
        print(f"Direct load failed: {e}")

    # Strategy 2: Zip Extraction (For Keras 3 model on TF 2.10)
    if zipfile.is_zipfile(model_path):
        print("Detected .keras Zip format. Attempting to extract weights for legacy loading...")
        try:
            with zipfile.ZipFile(model_path, 'r') as z:
                # Extract model.weights.h5 to a temp location
                if 'model.weights.h5' in z.namelist():
                    temp_weights = os.path.join(checkpoint_dir, 'temp_weights.h5')
                    with z.open('model.weights.h5') as zf, open(temp_weights, 'wb') as f:
                        shutil.copyfileobj(zf, f)
                    
                    print(f"Extracted weights to {temp_weights}")
                    
                    # Build architecture
                    print("Building fresh model architecture...")
                    model = build_model()
                    
                    # Load weights
                    print("Loading weights...")
                    try:
                        # Attempt standard/topological load
                        model.load_weights(temp_weights)
                        print("Weights loaded successfully (Topological)!")
                    except Exception as e:
                        print(f"Topological load failed: {e}")
                        try:
                            # Attempt load by name
                            model.load_weights(temp_weights, by_name=True)
                            print("Weights loaded successfully (By Name)!")
                        except Exception as e2:
                            print(f"Load by name failed: {e2}")
                            print("Attempting manual Keras 3 extracted weight loading...")
                            import h5py
                            
                            try:
                                with h5py.File(temp_weights, 'r') as f:
                                    if 'layers' not in f:
                                        raise ValueError("No 'layers' group in HDF5")
                                    
                                    layers_group = f['layers']
                                    loaded_count = 0
                                    
                                    for layer in model.layers:
                                        if layer.name in layers_group:
                                            g = layers_group[layer.name]
                                            if 'vars' in g:
                                                vars_group = g['vars']
                                                # vars are usually named "0", "1", "2"...
                                                # We need to sort them numerically
                                                weight_names = sorted(vars_group.keys(), key=lambda x: int(x))
                                                weights = [np.array(vars_group[wn]) for wn in weight_names]
                                                
                                                try:
                                                    layer.set_weights(weights)
                                                    print(f"Loaded {layer.name}")
                                                    loaded_count += 1
                                                except Exception as ex:
                                                    print(f"Shape mismatch for {layer.name}: {ex}")
                                            else:
                                                print(f"No vars for {layer.name}")
                                        else:
                                            print(f"Layer {layer.name} not found in file")
                                    
                                    if loaded_count > 0:
                                        print(f"Manually loaded {loaded_count} layers.")
                                    else:
                                        raise ValueError("No layers matched.")
                            except Exception as e3:
                                print(f"Manual loading failed: {e3}")
                                import traceback
                                traceback.print_exc()
                                return None

                    # Clean up


                    os.remove(temp_weights)
                    return model
                else:
                    print("Could not find model.weights.h5 in the zip.")
        except Exception as e:
            print(f"Fallback extraction failed: {e}")
            import traceback
            traceback.print_exc()
            
    return None

def create_side_by_side_video(sample_idx, x_input, y_target, preds, save_path):
    """
    Creates a side-by-side comparison video (GIF).
    Left: Ground Truth
    Right: Model Prediction
    Hides the last frame (19th) to avoid artifacts.
    Showing Frames 0 to 18.
    """
    # x_input: (19, 64, 64, 1) -> Frames 0..18
    # y_target: (19, 64, 64, 1) -> Frames 1..19
    # preds:    (19, 64, 64, 1) -> Frames 1..19
    
    # We construct sequences for Frames 0..18
    
    # Ground Truth: x_input contains 0..18 exactly.
    gt_sequence = x_input 
    
    # Prediction:
    # Frame 0: From Input (x_input[0]) because we don't predict Frame 0.
    # Frames 1..18: From preds[0..17]
    valid_preds = preds[0:18]
    frame_0 = x_input[0:1]
    pred_sequence = np.concatenate([frame_0, valid_preds], axis=0)
    
    num_frames = 19
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(top=0.85) 
    
    im_left = axes[0].imshow(gt_sequence[0, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title("Ground Truth (Real)", fontsize=14)
    axes[0].axis('off')
    
    im_right = axes[1].imshow(pred_sequence[0, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title("Model Prediction", fontsize=14, color='blue')
    axes[1].axis('off')
    
    title_text = plt.suptitle(f"Sample {sample_idx+1} - Frame 0", fontsize=16)
    
    def update(frame_idx):
        im_left.set_data(gt_sequence[frame_idx, :, :, 0])
        im_right.set_data(pred_sequence[frame_idx, :, :, 0])
        title_text.set_text(f"Sample {sample_idx+1} - Frame {frame_idx}")
        return [im_left, im_right, title_text]
        
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=250, blit=False)
    
    print(f"Saving comparison GIF to {save_path}...")
    ani.save(save_path, writer='pillow', fps=4, savefig_kwargs={'dpi': 150}) # Increased DPI for GIF frames
    plt.close()

def create_static_comparison_grid(model, dataset, save_path, num_samples=5):
    """
    Creates a high-quality static grid image comparing GT vs Preds.
    """
    print(f"Generating static comparison grid for {num_samples} samples...")
    
    indices_to_show = [0, 5, 10, 15, 18] # Ensure 18 is max index shown (19th frame skipped)
    num_cols = len(indices_to_show)
    
    fig, axes = plt.subplots(2 * num_samples, num_cols, figsize=(3 * num_cols, 4 * num_samples))
    
    for x_batch, y_batch in dataset.take(1):
        preds_batch = model.predict(x_batch, verbose=0)
        
        for i in range(num_samples):
            if i >= x_batch.shape[0]: break
            
            x = x_batch[i] 
            preds = preds_batch[i] 
            
            # Row 1: Ground Truth
            for col_idx, frame_idx in enumerate(indices_to_show):
                ax = axes[i*2, col_idx]
                ax.imshow(x[frame_idx, :, :, 0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"Frame {frame_idx}", fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(f"Sample {i+1}\nReal", fontsize=12)
            
            # Row 2: Prediction
            for col_idx, frame_idx in enumerate(indices_to_show):
                ax = axes[i*2 + 1, col_idx]
                
                if frame_idx == 0:
                    img = x[0, :, :, 0]
                else:
                    # predictions are shifted by 1. Pred for frame k is at index k-1
                    img = preds[frame_idx - 1, :, :, 0]
                
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.axis('off')
                if col_idx == 0:
                    ax.set_ylabel(f"Sample {i+1}\nPred", fontsize=12, color='blue')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Static grid saved to {save_path}")

def main():
    print("=== Generating Presentation Assets ===")
    
    # Robust Load
    model = robust_load_model()
    if model is None: 
        print("CRITICAL ERROR: Model could not be loaded.")
        return

    # Load Data (Batch size 8)
    dataset = get_dataset(batch_size=8, augment=False)
    
    presentation_dir = os.path.join(RESULTS_DIR, 'presentation_assets')
    os.makedirs(presentation_dir, exist_ok=True)
    
    # 1. Generate GIFs
    print("\n--- Generating GIFs ---")
    num_gifs = 3
    for i, (x_batch, y_batch) in enumerate(dataset.take(1)):
        preds_batch = model.predict(x_batch, verbose=0)
        
        for j in range(num_gifs):
            if j >= x_batch.shape[0]: break
            save_path = os.path.join(presentation_dir, f'comparison_video_{j+1}.gif')
            create_side_by_side_video(j, x_batch[j].numpy(), y_batch[j].numpy(), preds_batch[j], save_path)
            
    # 2. Generate Static Grid
    print("\n--- Generating Static Grid ---")
    save_path_grid = os.path.join(presentation_dir, 'comparison_grid.png')
    create_static_comparison_grid(model, dataset, save_path_grid, num_samples=5)

    # 2.5 Copy existing plots
    print("\n--- Copying Existing Plots ---")
    plots_to_copy = ['loss_curve.png', 'evaluation_results.png']
    for plot_name in plots_to_copy:
        src = os.path.join(RESULTS_DIR, plot_name)
        dst = os.path.join(presentation_dir, plot_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {plot_name} to {presentation_dir}")
        else:
            print(f"Warning: {plot_name} not found in {RESULTS_DIR}")
    
    # 3. Calculate and Print Metrics
    print("\n--- Calculating Final Metrics ---")
    psnr, ssim, mse, mae = calculate_average_metrics(model, dataset, num_batches=10)
    
    metrics_path = os.path.join(presentation_dir, 'metrics_report.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== Model Evaluation Metrics ===\n")
        f.write(f"PSNR: {psnr:.4f}\n")
        f.write(f"SSIM: {ssim:.4f}\n")
        f.write(f"MSE:  {mse:.6f}\n")
        f.write(f"MAE:  {mae:.6f}\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    # 4. Generate Attention Heatmap
    print("\n--- Generating Attention Heatmap ---")
    try:
        generate_attention_heatmap(model, dataset, presentation_dir)
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nSUCCESS! All assets generated in: {presentation_dir}")

def generate_attention_heatmap(model, dataset, save_dir):
    """
    Generates a heatmap visualization based on the activations of the last ConvLSTM layer.
    Since the model structure in model.py strictly uses ConvLSTM2D without the explicit 
    SpatialAttention layer from layers.py, we visualize the feature activations 
    of the deepest recurrent layer to show 'where the model is looking'.
    """
    import cv2
    
    # Identify the last ConvLSTM2D layer
    target_layer = None
    # Iterate backwards to find the last ConvLSTM2D
    for layer in reversed(model.layers):
        if 'conv_lstm2d' in layer.name:
            target_layer = layer
            break
            
    if target_layer is None:
        print("Warning: No ConvLSTM2D layer found for heatmap visualization.")
        return

    print(f"Visualizing activations for layer: {target_layer.name} as proxy for Attention.")
    
    # Create a sub-model to get intermediate output
    # Input: Model Input
    # Output: Target Layer Output
    activation_model = tf.keras.Model(inputs=model.input, outputs=target_layer.output)

    # Get a sample
    for x_batch, y_batch in dataset.take(1):
        # Shape: (B, T, H, W, C) -> (8, 19, 64, 64, 1)
        # Predict activations
        activations = activation_model.predict(x_batch, verbose=0) 
        # Output shape of ConvLSTM2D with return_sequences=True: (B, T, H, W, Filters)
        # e.g., (8, 19, 64, 64, 64)
        
        # We focus on the last timestep of the first sample
        sample_acts = activations[0, -1, :, :, :] # (64, 64, 64)
        
        # Summarize filters to create a 2D map
        # Mean across filters
        heatmap_np = np.mean(sample_acts, axis=-1) # (64, 64)
        
        # Normalize to 0-1
        heatmap_np = np.maximum(heatmap_np, 0) # ReLU-like, just in case
        heatmap_np /= (np.max(heatmap_np) + 1e-7)
        
        # Original Input Frame (Last frame)
        input_frame = x_batch[0, -1, :, :, 0].numpy()
        
        # Post-process for display
        heatmap_norm = np.uint8(255 * heatmap_np)
        input_norm = np.uint8(255 * input_frame)
        
        # Colorize
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        input_bgr = cv2.cvtColor(input_norm, cv2.COLOR_GRAY2BGR)
        
        # Overlay
        overlay = cv2.addWeighted(input_bgr, 0.6, heatmap_color, 0.4, 0)
        
        # Zoom for presentation (x4)
        input_large = cv2.resize(input_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        heatmap_large = cv2.resize(heatmap_color, (256, 256), interpolation=cv2.INTER_NEAREST)
        overlay_large = cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Create Side-by-Side
        final_img = np.hstack([input_large, heatmap_large, overlay_large])
        
        save_path = os.path.join(save_dir, 'attention_heatmap.png')
        cv2.imwrite(save_path, final_img)
        print(f"Heatmap saved to {save_path}")
        return

if __name__ == "__main__":
    main()

