"""DS-CNN-S — published TinyML keyword architecture (Hello Edge / ARM).

The letters of the keyword are not in this file. They live in the training
data and in shared.config.KEYWORD. Swap the word by retraining.
"""

from __future__ import annotations

from shared.feature_spec import FEATURE_SHAPE

N_CLASSES = 3  # keyword, unknown, silence


def build_dscnn_s(n_classes: int = N_CLASSES, input_shape=FEATURE_SHAPE):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=input_shape, name="mfcc")
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for _ in range(4):
        # Explicit pad + VALID depthwise: same geometry as SAME 3×3, but the
        # converted graph is PAD + DEPTHWISE_CONV which ESP-NN can take the
        # S3 SIMD path on. SAME-padded depthwise was the 36 ms fallback.
        x = layers.ZeroPadding2D(((1, 1), (1, 1)))(x)
        x = layers.DepthwiseConv2D((3, 3), padding="valid", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, name="logits")(x)
    x = layers.Softmax(name="probs")(x)
    return keras.Model(inp, x, name="dscnn_s")
