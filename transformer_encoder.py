"""
    Transformer Encoder:
        contains stack of N identical layers where each layer consist of two sub-layers.
        i. Multihead Attention (first sub-layer)
        2. Feed forward layer (second sub-layer)

        The feed forward layer contains two linear transformation layers and ReLU
        activation in between. out dimensionality of first transformation layer dff=2048,
        while the second layer producr output of dimensionality dmodel=512.


        Layer normalization
"""

from scaled_dot_product_attention import DotProductAttention
from multi_head import MultiHeadAttention
from tensorflow import keras
from keras.layers import Dense, ReLU, LayerNormalization


# feed forward layer
def feedForward(x, d_ff, d_model):
    # initialize layers for feed-forward
    layer1 = Dense(d_ff)
    layer2 = Dense(d_model)
    activation = ReLU()

    x_layer1 = layer1(x)
    x_relu = ReLU(x_layer1)
    x_layer2 = layer2(x_relu)

    return x_layer2


# Add and normalization layer
def addAndNorm(x_in, x_out):
    layer_norm = LayerNormalization()

    add = x_in + x_out

    return layer_norm(add)
