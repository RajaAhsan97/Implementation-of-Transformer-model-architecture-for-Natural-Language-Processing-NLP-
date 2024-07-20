"""
    Multi head attention 
"""

import numpy as np
from scaled_dot_product_attention import DotProductAttention
from tensorflow import keras, reshape, shape, transpose
from keras.layers import Dense

def split_heads(Tensor, h):
    x = reshape(Tensor, shape=(shape(Tensor)[0], shape(Tensor)[1], h, -1))
    x = transpose(x, perm=(0,2,1,3))
    return x

def concat_heads(x , dk):
    x = transpose(x, perm=(0,2,1,3))
    x = reshape(x, shape=(shape(x)[0], shape(x)[1], dk))
    return x

def MultiHeadAttention(q, k, v, dk, dv, dmodel, h):
    # set Dense layers for processing query, key and value
    Layer_q = Dense(dk)
    Layer_k = Dense(dk)
    Layer_v = Dense(dv)
    Layer_final = Dense(dmodel)

    # process q, v and k through Dense layer
    q_dense = Layer_q(q)
    k_dense = Layer_k(k)
    v_dense = Layer_v(v)

    # reshape q, k and v into 8 heads
    q = split_heads(q_dense, h)
    k = split_heads(k_dense, h)
    v = split_heads(v_dense, h)

    # process q, k and v through scaled_dot_product_attention
    SDPA_out = DotProductAttention(q, k, v, dk)

    # concatenate the output
    concat = concat_heads(SDPA_out, dk)

    # process through final layer
    return Layer_final(concat)    

### set no. of heads for query, key and value (typical value is 8)
##heads = 8
### set input sequence length
##sequence_length = 5
### dimensionality for query and key
##dk = 64
### dimensionality for values
##dv = 64
###
##d_model = 512
### batch size
##batch_size = 64
##
##query = np.random.random((batch_size, sequence_length, dk))
##key = np.random.random((batch_size, sequence_length, dk))
##value = np.random.random((batch_size, sequence_length, dv))
##
##MHA_out = MultiHeadAttention(query, key, value, dk, dv, d_model, heads)
