"""PoolFormer-style hierarchical encoder.

A lightweight Vision-Transformer backbone that replaces self-attention with
parameter-free spatial pooling (token mixing). It outputs a four-stage feature
pyramid that is consumed by the depth and de-weather decoder heads.
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class DropPath(tf.keras.layers.Layer):
    """Stochastic depth: randomly drops the residual branch during training."""

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop_path": self.drop_path,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class MixMLP(tf.keras.layers.Layer):
    """Feed-forward block with a depthwise conv inserted between two dense layers."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop_rate=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.drop_rate = drop_rate

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features)

        self.dwconv = tf.keras.layers.SeparableConv2D(
            filters=hidden_features,
            kernel_size=3,
            strides=1,
            padding="same",
            groups=hidden_features,
            depthwise_initializer=tf.keras.initializers.HeUniform(),
            pointwise_initializer=tf.keras.initializers.HeUniform(),
        )
        self.act = tf.keras.layers.Activation("gelu")
        self.fc2 = tf.keras.layers.Dense(out_features)

        self.drop = tf.keras.layers.SpatialDropout2D(self.drop_rate)

    def call(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "drop_rate": self.drop_rate,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class TokenMixer(tf.keras.layers.Layer):
    """Attention-free token mixer: average pooling minus the identity."""

    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pool = tf.keras.layers.AveragePooling2D(kernel_size, 1, padding='same')
    
    def call(self, x):
        x = self.pool(x) - x
        return x
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
            }
        )
        return config    

@tf.keras.utils.register_keras_serializable()
class Transformer_Block(tf.keras.layers.Layer):
    """One PoolFormer block: token mixing + MixMLP, each a residual with layer scale."""

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        drop=0,
        attn_drop=0,
        drop_path_rate=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path_rate = drop_path_rate


        self.norm1 = tf.keras.layers.GroupNormalization(1, epsilon=1e-5)

        self.attn = TokenMixer(kernel_size=3)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else tf.keras.layers.Layer()
        )
        self.norm2 = tf.keras.layers.GroupNormalization(1, epsilon=1e-5)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop_rate=drop,
        )
        self.layer_scale1 = tf.Variable(1e-5 * tf.ones(dim), True)
        self.layer_scale2 = tf.Variable(1e-5 * tf.ones(dim), True)


    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale1)
        x = x + self.drop_path(self.mlp(self.norm2(x)) * self.layer_scale2)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "mlp_ratio": self.mlp_ratio,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path_rate": self.drop_path_rate,

            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class OverlapPatchEmbed(tf.keras.layers.Layer):
    """Overlapping patch embedding that downsamples and projects the feature map."""

    def __init__(
        self, patch_size=7, stride=4, filters=768, **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.filters = filters

        self.pad = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=patch_size,
            strides=stride,
            padding="VALID",
            name='proj',
        )
        self.norm = tf.keras.layers.GroupNormalization(1, epsilon=1e-5)


    def call(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "filters": self.filters,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class Poolformer(tf.keras.layers.Layer):
    """Four-stage PoolFormer encoder producing a hierarchical feature pyramid."""

    def __init__(
        self,
        embed_dims=[32, 64, 128, 256],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        depths=[3, 4, 6, 3],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.depths = depths
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7,
            stride=4,
            filters=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            filters=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            filters=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            filters=embed_dims[3],
        )

        dpr = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = [
            Transformer_Block(
                dim=embed_dims[0],
                mlp_ratio=mlp_ratios[0],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
            )
            for i in range(depths[0])
        ]
        self.norm1 = tf.keras.layers.GroupNormalization(1, epsilon=1e-5)


        cur += depths[0]
        self.block2 = [
            Transformer_Block(
                dim=embed_dims[1],
                mlp_ratio=mlp_ratios[1],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
            )
            for i in range(depths[1])
        ]
        self.norm2 = tf.keras.layers.GroupNormalization(1, epsilon=1e-5)

        cur += depths[1]
        self.block3 = [
            Transformer_Block(
                dim=embed_dims[2],
                mlp_ratio=mlp_ratios[2],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
            )
            for i in range(depths[2])
        ]
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)



        cur += depths[2]
        self.block4 = [
            Transformer_Block(
                dim=embed_dims[3],
                mlp_ratio=mlp_ratios[3],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
            )
            for i in range(depths[3])
        ]
        self.norm4 = tf.keras.layers.LayerNormalization(epsilon=1e-5)



    def call_features(self, x):
        B = tf.shape(x)[0]
        outs = []

        # stage 1
        x = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x)
        x = self.norm1(x)
        outs.append(x)

        # stage 2
        x = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x)
        x = self.norm2(x)
        outs.append(x)

        # stage 3
        x = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x)
        x = self.norm3(x)
        outs.append(x)

        # stage 4
        x = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x)
        x = self.norm4(x)
        outs.append(x)

        return outs

    @tf.function
    def call(self, x):
        x = self.call_features(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "mlp_ratios": self.mlp_ratios,
                "depths": self.depths,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )
        return config
