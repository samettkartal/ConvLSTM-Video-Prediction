import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional

from config import CHECKPOINT_DIR
from layers import ChannelAttention, SpatialAttention
from losses import TotalLoss, PerceptualLoss, GradientDifferenceLoss

def calculate_psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def calculate_ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate Structural Similarity Index (SSIM).
    """
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def calculate_average_metrics(model: tf.keras.Model, dataset: tf.data.Dataset, num_batches: int = 20) -> Tuple[float, float]:
    """
    Calculate average PSNR and SSIM on the dataset.
    
    Args:
        model: Trained model.
        dataset: tf.data.Dataset object.
        num_batches: Number of batches to evaluate.
        
    Returns:
        Tuple of (average_psnr, average_ssim).
    """
    print("Evaluating metrics on test set...")
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    
    mse_fn = tf.keras.losses.MeanSquaredError()
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    
    for x, y in dataset.take(num_batches):
        preds = model.predict(x, verbose=0)
        
        # Reshape for metrics: (B*T, H, W, C)
        b, t, h, w, c = tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], tf.shape(y)[3], tf.shape(y)[4]
        y_reshaped = tf.reshape(y, (b*t, h, w, c))
        preds_reshaped = tf.reshape(preds, (b*t, h, w, c))
        
        # Cast to float32
        y_reshaped = tf.cast(y_reshaped, tf.float32)
        preds_reshaped = tf.cast(preds_reshaped, tf.float32)
        
        batch_psnr = tf.reduce_mean(calculate_psnr(y_reshaped, preds_reshaped))
        batch_ssim = tf.reduce_mean(calculate_ssim(y_reshaped, preds_reshaped))
        batch_mse = mse_fn(y_reshaped, preds_reshaped)
        batch_mae = mae_fn(y_reshaped, preds_reshaped)
        
        total_psnr += batch_psnr
        total_ssim += batch_ssim
        total_mse += batch_mse
        total_mae += batch_mae
        count += 1
        
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_psnr = float(total_psnr / count)
    avg_ssim = float(total_ssim / count)
    avg_mse = float(total_mse / count)
    avg_mae = float(total_mae / count)
    
    return avg_psnr, avg_ssim, avg_mse, avg_mae

def visualize_predictions(model: tf.keras.Model, dataset: tf.data.Dataset, save_path: str, num_samples: int = 5):
    """
    Visualize test results: Input (Last 5), Target (First 5), Prediction (First 5).
    """
    print(f"Generating visualization for {num_samples} samples...")
    for x, y in dataset.take(1):
        preds = model.predict(x, verbose=0)
        
        # Create a large figure
        # Rows: 3 (Input, Target, Pred) per sample
        # Cols: 5 (Frames)
        
        # Note: Previous implementation used flat subplots, new one uses subplots grid
        # We'll stick to a clean grid layout
        
        fig, axes = plt.subplots(3 * num_samples, 6, figsize=(15, 6 * num_samples))
        
        for i in range(num_samples):
            if i >= x.shape[0]: break
            
            # Row index for this sample
            row_start = i * 3
            
            # 1. Input Frames (Last 5 frames of input context)
            for t in range(5):
                ax = axes[row_start, t]
                ax.imshow(x[i, 5+t, :, :, 0], cmap='gray')
                if t == 0: ax.set_ylabel(f'Sample {i+1}\nInputs', fontsize=12)
                ax.set_title(f't={6+t}')
                ax.axis('off')
            axes[row_start, 5].axis('off') # Spacer

            # 2. Ground Truth Frames (First 5 frames of prediction target)
            for t in range(5):
                ax = axes[row_start + 1, t]
                ax.imshow(y[i, t, :, :, 0], cmap='gray')
                if t == 0: ax.set_ylabel(f'Targets', fontsize=12)
                ax.set_title(f't={11+t}')
                ax.axis('off')
            axes[row_start + 1, 5].axis('off') # Spacer

            # 3. Predicted Frames
            for t in range(5):
                ax = axes[row_start + 2, t]
                ax.imshow(preds[i, t, :, :, 0], cmap='gray')
                if t == 0: ax.set_ylabel(f'Preds', fontsize=12)
                ax.set_title(f't={11+t}')
                ax.axis('off')
            axes[row_start + 2, 5].axis('off') # Spacer

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")

def load_trained_model(checkpoint_dir: str = CHECKPOINT_DIR) -> Optional[tf.keras.Model]:
    """
    Load the trained model with custom objects.
    """
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'TotalLoss': TotalLoss,
        'PerceptualLoss': PerceptualLoss,
        'GradientDifferenceLoss': GradientDifferenceLoss
    }
    
    model_path = os.path.join(checkpoint_dir, 'model_best.keras')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Checking for final model...")
        model_path = os.path.join(checkpoint_dir, 'model_final.keras')
        
    if not os.path.exists(model_path):
        print("No trained model found! Please run train.py first.")
        return None

    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_loss_plot(history: tf.keras.callbacks.History, save_path: str):
    """
    Save training loss plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
