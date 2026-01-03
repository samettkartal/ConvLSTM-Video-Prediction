import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from data_loader import get_train_val_datasets
import matplotlib.animation as animation

def create_comparison_gif(x, y_true, y_pred_user, y_pred_ref, save_path='comparison.gif'):
    """
    Creates a side-by-side comparison GIF:
    Left: Ground Truth (Target)
    Middle: Reference Model (Keras-IO)
    Right: User Model
    """
    # x shape: (T, H, W, C) - Input frames
    # y_true shape: (T, H, W, C) - Target frames (Next frames)
    # y_pred_* shape: (T, H, W, C)
    
    # We want to show the GROUND TRUTH VIDEO (which is x + last y?)
    # Or just show the Target sequence (y_true) vs Predictions?
    # The user said: "ilk başta gerçek veri ortada uzantıdaki model en sağda da model tahmini olsun"
    # "Gerçek veri" usually means the Ground Truth (Target) or the Input?
    # Usually in Next-Frame prediction, we compare the Predicted Next Frame vs the Actual Next Frame.
    # So we compare y_true vs y_pred.
    
    # Ensure numpy
    y_true = y_true.numpy() if hasattr(y_true, 'numpy') else y_true
    y_pred_user = y_pred_user.numpy() if hasattr(y_pred_user, 'numpy') else y_pred_user
    y_pred_ref = y_pred_ref.numpy() if hasattr(y_pred_ref, 'numpy') else y_pred_ref
    
    # Remove batch dim if present (take first sample)
    if len(y_true.shape) == 5: # (B, T, H, W, C)
        y_true = y_true[0]
        y_pred_user = y_pred_user[0]
        y_pred_ref = y_pred_ref[0]
        
    # Frames count
    frames = y_true.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def update(i):
        for ax in axes: ax.clear()
        
        # Ground Truth
        axes[0].imshow(y_true[i, :, :, 0], cmap='gray')
        axes[0].set_title(f"Ground Truth (Frame {i+1})")
        axes[0].axis('off')
        
        # Ref Model
        # Handle case where ref model might output different length or shape?
        # We assume it outputs same length for now.
        if i < len(y_pred_ref):
            axes[1].imshow(y_pred_ref[i, :, :, 0], cmap='gray')
            axes[1].set_title(f"Ref Model (Keras-IO)")
        else:
            axes[1].text(0.5, 0.5, "No Frame", ha='center')
        axes[1].axis('off')

        # User Model
        if i < len(y_pred_user):
            axes[2].imshow(y_pred_user[i, :, :, 0], cmap='gray')
            axes[2].set_title(f"User Model")
        else:
            axes[2].text(0.5, 0.5, "No Frame", ha='center')
        axes[2].axis('off')
        
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
    ani.save(save_path, writer='pillow')
    print(f"Saved comparison to {save_path}")

def main():
    print("Setting up environment...")
    
    # 1. Load Data
    print("Loading Dataset (Moving MNIST)...")
    # Using the local data_loader logic
    # It downloads/loads from tfds
    train_ds, val_ds = get_train_val_datasets(batch_size=1)
    
    # Take one sample for visualization
    print("Sampling a batch...")
    for batch_x, batch_y in val_ds.take(1):
        sample_x = batch_x # (1, 19, 64, 64, 1)
        sample_y = batch_y # (1, 19, 64, 64, 1)
        break
        
    # 2. Load User Model
    print("Loading User Model...")
    model_path = "model_final.keras"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    try:
        # Load without compiling to avoid custom object needs for Loss/Optimizer
        user_model = tf.keras.models.load_model(model_path, compile=False)
        print("User Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load user model: {e}")
        # Fallback: Try building and loading weights?
        # from model import build_model
        # user_model = build_model()
        # user_model.load_weights(model_path) # if it was h5 weights
        return

    # 3. Load Reference Model
    print("Loading Reference Model from Hugging Face (keras-io/conv-lstm)...")
    try:
        ref_model = from_pretrained_keras("keras-io/conv-lstm")
        print("Reference Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load reference model: {e}")
        return

    # 4. Predict
    print("Running Predictions...")
    
    # User Prediction
    print("Predicting with User Model...")
    user_pred = user_model.predict(sample_x)
    
    # Ref Prediction
    print("Predicting with Reference Model...")
    # Reference model expectation checking
    
    # The reference model might expect dynamic shape or fixed.
    # ConvLSTM usually handles dynamic T. 
    # Let's try passing the sample_x directly.
    try:
        ref_pred = ref_model.predict(sample_x)
    except Exception as e:
        print(f"Reference prediction failed: {e}")
        # Try resizing?
        # sample_x is 64x64. Ref model is likely 64x64.
        print("Detailed error:", e)
        return

    # 5. Visualize
    print("Generating Comparison GIF...")
    create_comparison_gif(sample_x, sample_y, user_pred, ref_pred, save_path="benchmark_comparison.gif")
    
    print("Done! Download 'benchmark_comparison.gif' to view.")

if __name__ == "__main__":
    main()
