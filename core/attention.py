import numpy as np
from transformer.encoder.self_attention.softmax import row_softmax


def scaled_dot_product_attention(Q, K, V, mask=None):

    dk = K.shape[-1]

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dk)

    if mask is not None:
        scores = scores + mask

    attention_weights = row_softmax(scores)

    output = attention_weights @ V

    return output, attention_weights