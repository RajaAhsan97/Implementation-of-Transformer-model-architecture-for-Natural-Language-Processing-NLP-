"""
    Scaled Dot Product Attention:
    1. (STAGE 1)   Matmul    --->  q and k
    2. (STAGE 2)   Scaling   --->  o/p STAGE1/sqrt(d)  where, d=dk=dv
       (STAGE )    Masking   --->  optional
    3. (STAGE 3)   Softmax   --->  process o/p of STAGE 2
    4. (STAGE 4)   Matmul    --->  o/p of STAGE 3 and v

    First try with random query, key, value.

    Then, implement query tokenization and embedding for obtaining query, key and value vectors
"""

import numpy as np
from tensorflow import matmul, nn

def DotProductAttention(q, k, v, d):
    # STAGE 1
    scores = matmul(q, k, transpose_a=False, transpose_b=True)

    # STAGE 2
    scaled_scores = scores / np.sqrt(d)

    # STAGE 3
    Softmax_out = nn.softmax(scaled_scores)

    # STAGE 4
    final_out = matmul(Softmax_out, v)

    return final_out


### set input sequence length
##sequence_length = 5
### dimensionality for query and key
##dk = 64
### dimensionality for values
##dv = 64
### batch size
##batch_size = 64
##
##query = np.random.random((batch_size, sequence_length, dk))
##key = np.random.random((batch_size, sequence_length, dk))
##value = np.random.random((batch_size, sequence_length, dv))
##
##out = DotProductAttention(query, key, value, dk)
