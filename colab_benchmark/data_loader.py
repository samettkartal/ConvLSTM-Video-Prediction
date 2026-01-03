import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from config import BATCH_SIZE, IMG_SIZE, SEQ_LEN, INPUT_LEN, PRED_LEN

def load_moving_mnist():
    """
    Load MovingMNIST dataset from tensorflow_datasets.
    """
    # tfds.load returns a tf.data.Dataset
    ds = tfds.load('moving_mnist', split='test', shuffle_files=False)
    return ds

def preprocess_data(features):
    """
    Normalize and split data into input and target.
    """
    video = features['image_sequence']
    video = tf.cast(video, tf.float32) / 255.0
    
    # Resize if needed (MovingMNIST is 64x64, so this might be redundant but safe)
    # video = tf.image.resize(video, IMG_SIZE)
    
    # Split into shifted sequences for Next-Frame Prediction
    # x: Frames 0 to 18 (Input)
    # y: Frames 1 to 19 (Target: Next frame for each input frame)
    x = video[:-1]
    y = video[1:]
    
    # Ensure shape is (T, H, W, C)
    x = tf.reshape(x, (19, IMG_SIZE[0], IMG_SIZE[1], 1))
    y = tf.reshape(y, (19, IMG_SIZE[0], IMG_SIZE[1], 1))
    
    return x, y

def augment_data(x, y):
    """
    Apply data augmentation to the video sequence.
    Augmentations must be consistent across time steps.
    """
    # Concatenate x and y to apply same augmentation
    combined = tf.concat([x, y], axis=0)
    
    # Random Rotation (0, 90, 180, 270 degrees)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    
    # Random Flip Left-Right
    combined = tf.image.random_flip_left_right(combined)
    
    # Random Flip Up-Down
    combined = tf.image.random_flip_up_down(combined)
    
    # Add Gaussian Noise
    noise = tf.random.normal(shape=tf.shape(combined), mean=0.0, stddev=0.01)
    combined = combined + noise
    combined = tf.clip_by_value(combined, 0.0, 1.0)
    
    # Split back
    x = combined[:INPUT_LEN]
    y = combined[INPUT_LEN:]
    
    return x, y

def get_train_val_datasets(batch_size=BATCH_SIZE, val_split=1000):
    """
    Create Train and Validation datasets with proper splitting and processing.
    """
    ds = load_moving_mnist()
    
    # Map preprocessing (Normalization, Reshape)
    ds = ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split into Train and Val
    # Note: MovingMNIST is small (10k), so we can just take/skip.
    # We take the FIRST 'val_split' for validation to ensure it's fixed (if ds is deterministic enough)
    # However, tfds.load with split='test' and shuffle_files=True might be random.
    # Ideally, we should set shuffle_files=False in load_moving_mnist for deterministic split.
    
    val_ds = ds.take(val_split)
    train_ds = ds.skip(val_split)
    
    # --- Prepare Validation Set ---
    # No Shuffle, No Augmentation
    val_ds = val_ds.batch(batch_size, drop_remainder=False) # Keep all val data
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # --- Prepare Training Set ---
    # Shuffle
    train_ds = train_ds.shuffle(1000)
    
    # Augment
    # CAUTION: Disabling augmentation for now as it causes T/H/W inconsistencies in some TF versions
    # train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    
    # Prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


# Kept for backward compatibility if needed, but updated to use new logic (returns only train-like)
def get_dataset(batch_size=BATCH_SIZE, augment=True):
    return get_train_val_datasets(batch_size, val_split=0)[0] # Return full as train
