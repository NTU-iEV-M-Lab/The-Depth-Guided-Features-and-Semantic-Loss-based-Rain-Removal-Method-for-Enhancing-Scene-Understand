"""De-weather decoder head with Depth-Guided Spatial Feature Transform (DG-SFT).

This head decodes the PoolFormer feature pyramid into a restored (rain- and
fog-free) image. Its key component is the DG-SFT block: the predicted depth map
modulates the decoded features through a learned spatial scale and shift,
following the distance-dependent rain/haze model so that distant, fog-degraded
regions are corrected more strongly.
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class WeatherHead_v10(tf.keras.layers.Layer):
    """De-weather decoder: restores a clean image, depth-guided via DG-SFT."""

    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.hidden_layers = []          
        self.sft_layers1 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                32, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.SeparableConv2D(
                1, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.Activation('gelu')
        ])
        self.sft_layers2 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                32, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.SeparableConv2D(
                1, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.Activation('gelu')
        ])
        
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dim // 2
            ),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Activation('gelu'),

            tf.keras.layers.SeparableConv2D(
                hidden_dim // 2, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation('gelu')
        ])
        self.dense1 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dim // 2
            ),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.UpSampling2D(
                (2, 2), interpolation='bilinear'
            ),
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                hidden_dim // 4, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation('gelu')
        ])
        self.dense2 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dim // 4
            ),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.UpSampling2D(
                (4, 4), interpolation='bilinear'
            ),
        ])
        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                hidden_dim // 8, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation('gelu')
        ])
        self.dense3 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dim // 8
            ),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.UpSampling2D(
                (8, 8), interpolation='bilinear'
            ),
        ])
        self.conv4 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                32, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.UpSampling2D(
                (4, 4), interpolation='bilinear'
            ),
        ])
        self.rain_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                32
            ),
            tf.keras.layers.GroupNormalization(1),
            tf.keras.layers.Activation('gelu'),
        ])
    
        
        self.linear_pred = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                16, 3, padding='same',
                pointwise_initializer=tf.initializers.HeUniform(),
                depthwise_initializer=tf.initializers.HeUniform()
            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.SpatialDropout2D(0.2),
            tf.keras.layers.SeparableConv2D(
                3, 3, padding='same',
                pointwise_initializer=tf.initializers.GlorotUniform(),
                depthwise_initializer=tf.initializers.GlorotUniform()
            ),
            tf.keras.layers.Activation('tanh'),

        ])

        

    @tf.function
    def call(self, encode_weather, depth, rain):
        """Decode the feature pyramid into a restored image, depth-guided by DG-SFT."""
        H = tf.shape(encode_weather[0])[1]
        W = tf.shape(encode_weather[0])[2]

        x = self.conv1(encode_weather[0])
        branch = self.dense1(encode_weather[1])
        x = self.conv2(tf.concat([x, branch], -1))
        branch = self.dense2(encode_weather[2])
        x = self.conv3(tf.concat([x, branch], -1))
        branch = self.dense3(encode_weather[3])
        x = tf.concat([x, branch], -1)
        image = self.conv4(x)

        # DG-SFT: the depth map predicts a spatial scale and shift that modulate
        # the decoded features (scale via multiply, shift via add).
        image = image * self.sft_layers1(depth)
        image = image + self.sft_layers2(depth)
        rain = self.rain_dense(rain)
        image = tf.concat([image, rain], -1)
        image = self.linear_pred(image)
        return image

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'hidden_dim': self.hidden_dim,
            }
        )
        return config