import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from losses import TotalLoss
from config import IMG_SIZE, INPUT_LEN, CHANNELS, LR, DROPOUT_RATE
from config import FILTERS, KERNEL_SIZES, L2_REG, BCE_WEIGHT, PERCEPTUAL_WEIGHT, GDL_WEIGHT, GDL_ALPHA, GRADIENT_CLIP_NORM

def build_model() -> Model:
    """
    Builds the Sequential ConvLSTM model based on Keras.io example.
    
    Architecture:
    1. Input (None, H, W, C)
    2. ConvLSTM2D (5x5) + BN + ReLU
    3. ConvLSTM2D (3x3) + BN + ReLU
    4. ConvLSTM2D (1x1) + BN + ReLU
    5. Conv3D (Output)
    """
    input_shape = (INPUT_LEN, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    inputs = Input(shape=input_shape)

    # Layer 1: Capture broad motion (5x5)
    x = layers.ConvLSTM2D(
        filters=FILTERS, 
        kernel_size=KERNEL_SIZES[0], 
        padding='same', 
        return_sequences=True, 
        activation='relu'
    )(inputs)
    x = layers.BatchNormalization()(x)

    # Layer 2: Refine details (3x3)
    x = layers.ConvLSTM2D(
        filters=FILTERS, 
        kernel_size=KERNEL_SIZES[1], 
        padding='same', 
        return_sequences=True, 
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)

    # Layer 3: Feature mixing (1x1)
    x = layers.ConvLSTM2D(
        filters=FILTERS, 
        kernel_size=KERNEL_SIZES[2], 
        padding='same', 
        return_sequences=True, 
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)

    # Output Layer: Spatiotemporal smoothing
    outputs = layers.Conv3D(
        filters=1, 
        kernel_size=(3, 3, 3), 
        activation='sigmoid', 
        padding='same'
    )(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=GRADIENT_CLIP_NORM)
    loss_fn = TotalLoss(
        bce_weight=BCE_WEIGHT,
        perceptual_weight=PERCEPTUAL_WEIGHT, 
        gdl_weight=GDL_WEIGHT,
        gdl_alpha=GDL_ALPHA
    )
    
    # Track accuracy (crucial for BCE) and MSE (for reference)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', 'mse'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
