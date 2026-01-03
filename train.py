import tensorflow as tf
import os
import matplotlib.pyplot as plt
from config import EPOCHS, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, PERCEPTUAL_WEIGHT, GDL_WEIGHT
from data_loader import get_dataset
from model import build_model
from utils import visualize_predictions, save_loss_plot, calculate_psnr, calculate_ssim


# Enable Mixed Precision
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

def main():
    # 1. Load Data
    print("Loading Data...")
    from data_loader import get_train_val_datasets
    train_dataset, val_dataset = get_train_val_datasets()
    
    # 2. Build Model
    print("Building Model...")
    model = build_model()
    model.summary()
    
    # 3. Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, 'model_best.keras'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # More aggressive reduction
        patience=3, # Wait less before reducing
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    
    
    class LossMonitorCallback(tf.keras.callbacks.Callback):
        """Monitor individual loss components during training"""
        def __init__(self):
            super().__init__()
            self.mse_fn = tf.keras.losses.MeanSquaredError()
            
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                # Sample a batch from validation to check loss components
                for x, y in val_dataset.take(1):
                    preds = self.model(x, training=False)
                    
                    # Calculate individual losses
                    # Note: We use the loss functions from the model instance if possible or re-instantiate
                    # Check if model.loss has attributes
                    if hasattr(self.model.loss, 'bce'):
                         bce_loss = self.model.loss.bce(y, preds) * self.model.loss.bce_weight
                         perc_loss = self.model.loss.perceptual(y, preds)
                         gdl_loss = self.model.loss.gdl(y, preds)
                         
                         print(f"\n[Epoch {epoch+1}] Loss Components:")
                         print(f"  BCE (Weighted): {bce_loss:.4f}")
                         print(f"  Perceptual: {perc_loss:.4f}")
                         print(f"  GDL: {gdl_loss:.4f}")
                         print(f"  Total Sum: {bce_loss + perc_loss + gdl_loss:.4f}")
                    else:
                         print("Could not access individual loss components.")
                    break
    
    loss_monitor_cb = LossMonitorCallback()
    
    class VisualizationCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(RESULTS_DIR, f'epoch_{epoch+1}.png')
                visualize_predictions(self.model, val_dataset, save_path)
                
    viz_cb = VisualizationCallback()
    
    # 4. Train
    print("Starting Training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb, loss_monitor_cb, viz_cb]
    )
    
    # 5. Save Final Model and Results
    model.save(os.path.join(CHECKPOINT_DIR, 'model_final.keras'))
    save_loss_plot(history, os.path.join(RESULTS_DIR, 'loss_curve.png'))
    
    print("Training Completed!")

if __name__ == "__main__":
    main()
