import numpy as np
from transformer.encoder.self_attention.softmax import row_softmax

def cross_attention(encoder_out, decoder_state):
    
    batch_size, seq_len_enc, d_model = encoder_out.shape
    _, seq_len_dec, _ = decoder_state.shape

    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)

    queries = decoder_state @ Wq
    keys = encoder_out @ Wk
    values = encoder_out @ Wv

    dk = keys.shape[-1]

    scores = queries @ keys.transpose(0, 2, 1) / np.sqrt(dk)

    attention_weights = row_softmax(scores)

    output = attention_weights @ values

    return output, attention_weights