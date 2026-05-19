"""Depth decoder head.

A lightweight CNN decoder that turns the PoolFormer feature pyramid into a
single-channel inverse-depth map. The predicted depth drives the Depth-Guided
Spatial Feature Transform (DG-SFT) in the de-weather head.
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class DepthHead_v7(tf.keras.layers.Layer):
    """CNN depth decoder: fuses the 4-stage feature pyramid into a depth map."""

    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.linear_pred = tf.keras.Sequential([
            tf.keras.layers.SpatialDropout2D(0.2),
            tf.keras.layers.SeparableConv2D(
                1, 3, padding='same',
            ),
            tf.keras.layers.Activation('sigmoid'),
        ])

        self.hidden_layers = []
        self.fusion_layers = []


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
        
            
    @tf.function
    def call(self, inputs):
        """Decode the 4-stage feature pyramid into a single-channel depth map."""
        H = tf.shape(inputs[0])[1]
        W = tf.shape(inputs[0])[2]

        x = self.conv1(inputs[0])
        branch = self.dense1(inputs[1])
        x = self.conv2(tf.concat([x, branch], -1))
        branch = self.dense2(inputs[2])
        x = self.conv3(tf.concat([x, branch], -1))
        branch = self.dense3(inputs[3])
        x = tf.concat([x, branch], -1)

        x = self.conv4(x)
        x = self.linear_pred(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
            }
        )
        return config