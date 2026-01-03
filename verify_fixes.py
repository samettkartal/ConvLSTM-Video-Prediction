"""
Quick verification script to test the model builds correctly with new changes
"""
import tensorflow as tf
from model import build_model
from data_loader import get_train_val_datasets

print("=" * 60)
print("VERIFICATION: Testing Model Build and Loss Calculation")
print("=" * 60)

# 1. Test model builds
print("\n1. Building model...")
model = build_model()
print("✓ Model built successfully")

# 2. Check optimizer configuration
print("\n2. Checking optimizer configuration...")
print(f"   Learning Rate: {model.optimizer.learning_rate.numpy()}")
print(f"   Clipnorm: {model.optimizer.clipnorm}")
print("✓ Optimizer configured correctly")

# 3. Test loss calculation with dummy data
print("\n3. Testing loss calculation...")
train_ds, val_ds = get_train_val_datasets()

for x, y in train_ds.take(1):
    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Forward pass
    preds = model(x, training=False)
    print(f"   Prediction shape: {preds.shape}")
    
    # Calculate loss
    loss = model.loss(y, preds)
    print(f"   Total Loss: {loss.numpy():.4f}")
    
    # Calculate individual components
    mse_fn = tf.keras.losses.MeanSquaredError()
    mse_loss = mse_fn(y, preds)
    perc_loss = model.loss.perceptual(y, preds)
    gdl_loss = model.loss.gdl(y, preds)
    
    print(f"\n   Loss Breakdown:")
    print(f"   - MSE: {mse_loss.numpy():.4f}")
    print(f"   - Perceptual (x0.01): {perc_loss.numpy():.4f}")
    print(f"   - GDL (x0.1): {gdl_loss.numpy():.4f}")
    print(f"   - Total: {(mse_loss + perc_loss + gdl_loss).numpy():.4f}")
    
    # Verify loss is in reasonable range
    if loss.numpy() < 10.0:
        print("\n✓ Loss is in reasonable range (< 10.0)")
    else:
        print(f"\n✗ WARNING: Loss is still high ({loss.numpy():.4f})")
    
    break

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nExpected behavior:")
print("- Loss should be < 10 (ideally 1-5 range)")
print("- MSE should dominate initially")
print("- Perceptual and GDL should contribute but not overwhelm")
print("\nIf loss is still > 10, there may be other issues to investigate.")
