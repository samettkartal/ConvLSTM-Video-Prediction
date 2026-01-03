import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from .config import IMG_SIZE, CHANNELS

class PerceptualLoss(tf.keras.losses.Loss):
    """
    Perceptual Loss using a pre-trained VGG16 model.
    Computes the L2 distance between feature maps of ground truth and prediction.
    """
    def __init__(self, layer_names=['block2_conv2'], weight=1.0, **kwargs):
        super(PerceptualLoss, self).__init__(**kwargs)
        self.weight = weight
        self.layer_names = layer_names
        self.vgg = self._build_vgg()

    def _build_vgg(self):
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in self.layer_names]
        return Model(inputs=vgg.input, outputs=outputs)

    def call(self, y_true, y_pred):
        # Preprocess inputs: grayscale to RGB, resize if necessary, normalize to VGG expectation
        # MovingMNIST is (B, T, H, W, 1), VGG expects (B, H, W, 3)
        # We calculate loss frame by frame or reshape
        
        # Cast to float32 to avoid overflow in loss calculation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        b, t, h, w, c = tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]
        
        y_true_reshaped = tf.reshape(y_true, (b * t, h, w, c))
        y_pred_reshaped = tf.reshape(y_pred, (b * t, h, w, c))
        
        # Convert to 3 channels
        y_true_rgb = tf.image.grayscale_to_rgb(y_true_reshaped)
        y_pred_rgb = tf.image.grayscale_to_rgb(y_pred_reshaped)
        
        # VGG expects inputs in [0, 255] centered, but we have [0, 1]. 
        # Simple scaling to [0, 255] is often enough for feature extraction.
        y_true_rgb = y_true_rgb * 255.0
        y_pred_rgb = y_pred_rgb * 255.0
        
        # Get features
        features_true = self.vgg(y_true_rgb)
        features_pred = self.vgg(y_pred_rgb)
        
        if not isinstance(features_true, list):
            features_true = [features_true]
            features_pred = [features_pred]
            
        loss = 0
        for f_true, f_pred in zip(features_true, features_pred):
            # Ensure features are float32 before squaring
            f_true = tf.cast(f_true, tf.float32)
            f_pred = tf.cast(f_pred, tf.float32)
            
            # Normalize features using L2 norm to prevent extremely high loss values
            # This brings feature magnitudes to a reasonable scale
            f_true_norm = tf.nn.l2_normalize(f_true, axis=-1)
            f_pred_norm = tf.nn.l2_normalize(f_pred, axis=-1)
            
            loss += tf.reduce_mean(tf.square(f_true_norm - f_pred_norm))
            
        return self.weight * loss

class GradientDifferenceLoss(tf.keras.losses.Loss):
    """
    Gradient Difference Loss (GDL).
    Penalizes differences in the image gradients to maintain sharpness.
    """
    def __init__(self, weight=1.0, alpha=2.0, **kwargs):  # Increased alpha to 2.0 for sharper edges
        super(GradientDifferenceLoss, self).__init__(**kwargs)
        self.weight = weight
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Cast to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate gradients in x and y directions
        # y_true, y_pred shape: (B, T, H, W, C)
        
        # Gradient X: |I(x+1, y) - I(x, y)|
        # Gradient Y: |I(x, y+1) - I(x, y)|
        
        # We can use tf.image.image_gradients but it works on 4D tensors.
        # Let's reshape to (B*T, H, W, C)
        b, t, h, w, c = tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]
        
        y_true_reshaped = tf.reshape(y_true, (b * t, h, w, c))
        y_pred_reshaped = tf.reshape(y_pred, (b * t, h, w, c))
        
        dy_true, dx_true = tf.image.image_gradients(y_true_reshaped)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred_reshaped)
        
        loss = tf.reduce_mean(tf.abs(tf.abs(dx_true) - tf.abs(dx_pred)) ** self.alpha + 
                              tf.abs(tf.abs(dy_true) - tf.abs(dy_pred)) ** self.alpha)
        
        return self.weight * loss

class TotalLoss(tf.keras.losses.Loss):
    """
    Combined Loss: BCE (Primary) + Perceptual + GDL
    BCE creates sharp binary pixels.
    GDL and Perceptual maintain structure and edges.
    """
    def __init__(self, bce_weight=1.0, perceptual_weight=0.1, gdl_weight=1.0, gdl_alpha=1.0, **kwargs):
        super(TotalLoss, self).__init__(**kwargs)
        self.bce_weight = bce_weight
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.perceptual = PerceptualLoss(weight=perceptual_weight)
        self.gdl = GradientDifferenceLoss(weight=gdl_weight, alpha=gdl_alpha)

    def call(self, y_true, y_pred):
        # Cast to float32 for safety
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        bce_loss = self.bce(y_true, y_pred)
        perc_loss = self.perceptual(y_true, y_pred)
        gdl_loss = self.gdl(y_true, y_pred)
        
        return (self.bce_weight * bce_loss) + perc_loss + gdl_loss
