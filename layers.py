import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    """
    Channel Attention Module.
    Models the relationship between channels of feature maps.
    Structure: GlobalAveragePooling -> Dense -> Dense -> Sigmoid
    """
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = layers.Dense(channel // self.ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_layer_two = layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        
        # Explicitly build sub-layers to ensure weights are created and can be loaded
        # Dense layer weights depend only on the last dimension
        self.shared_layer_one.build((None, channel))
        self.shared_layer_two.build((None, channel // self.ratio))
        
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        # Global Average Pooling
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        
        # MLP
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        
        # Sigmoid activation
        scale = tf.nn.sigmoid(avg_pool)
        scale = tf.cast(scale, inputs.dtype)
        
        return inputs * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module.
    Focuses on 'where' the important information is.
    Structure: Conv2D -> Sigmoid
    """
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(filters=1,
                                    kernel_size=self.kernel_size,
                                    strides=1,
                                    padding='same',
                                    activation='sigmoid',
                                    kernel_initializer='he_normal',
                                    use_bias=False)
        
        # Explicitly build sub-layer
        # Input to conv2d will be concatenation of avg_pool and max_pool (2 channels)
        if input_shape is not None:
            conv_input_shape = list(input_shape)
            conv_input_shape[-1] = 2 # 2 channels
            self.conv2d.build(tuple(conv_input_shape))
            
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # Average pooling and Max pooling across channels
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Conv2D -> Sigmoid
        scale = self.conv2d(concat)
        scale = tf.cast(scale, inputs.dtype)
        
        return inputs * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
