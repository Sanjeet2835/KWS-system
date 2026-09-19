"""DS-CNN-S with residual depthwise-separable blocks.

Designed for TinyML / ESP32-S3 deployment.
"""

from __future__ import annotations

from shared.feature_spec import FEATURE_SHAPE
import keras
from keras import layers

N_CLASSES = 3  # keyword, unknown, silence


def build_dscnn_s(
    n_classes: int = N_CLASSES,
    input_shape=FEATURE_SHAPE,
):


    inp = keras.Input(shape=input_shape, name="mfcc")

    # Stem
    x = layers.Conv2D(
        64,
        (10, 4),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inp)

    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Residual DS-CNN blocks
    for i in range(4):

        # Save residual
        residual = x

        # Depthwise convolution
        x = layers.ZeroPadding2D(
            ((1, 1), (1, 1)),
            name=f"block{i+1}_pad",
        )(x)

        x = layers.DepthwiseConv2D(
            (3, 3),
            padding="valid",
            use_bias=False,
            name=f"block{i+1}_dwconv",
        )(x)

        x = layers.BatchNormalization(
            name=f"block{i+1}_dw_bn",
        )(x)

        x = layers.ReLU(
            name=f"block{i+1}_dw_relu",
        )(x)

        # Pointwise projection
        x = layers.Conv2D(
            64,
            (1, 1),
            padding="same",
            use_bias=False,
            name=f"block{i+1}_pwconv",
        )(x)

        x = layers.BatchNormalization(
            name=f"block{i+1}_pw_bn",
        )(x)

        # Residual connection
        x = layers.Add(
            name=f"block{i+1}_add",
        )([x, residual])

        x = layers.ReLU(
            name=f"block{i+1}_out",
        )(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(
        name="global_avg_pool",
    )(x)

    x = layers.Dense(
        n_classes,
        name="logits",
    )(x)

    out = layers.Softmax(
        name="probs",
    )(x)

    return keras.Model(
        inp,
        out,
        name="dscnn_s_residual",
    )