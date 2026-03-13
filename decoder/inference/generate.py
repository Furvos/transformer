import numpy as np

VOCAB_SIZE = 10000

def generate_next_token(current_sequence, encoder_out):

    seq_len = len(current_sequence)
    d_model = encoder_out.shape[-1]

    decoder_state = np.random.randn(1, seq_len, d_model)

    W_vocab = np.random.randn(d_model, VOCAB_SIZE)

    last_token_vector = decoder_state[0, -1]

    logits = last_token_vector @ W_vocab

    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    return probs