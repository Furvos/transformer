import numpy as np
from transformer.decoder.causal_mask.causal_mask import create_causal_mask
from transformer.encoder.self_attention.softmax import row_softmax

def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

seq_len = 5
d_model = 4

# matrizes fictícias
queries = np.random.randn(seq_len, d_model)
keys = np.random.randn(seq_len, d_model)

scores = queries @ keys.T

print("Scores sem máscara:")
print(scores)

mask = create_causal_mask(seq_len)

masked_scores = scores + mask

print("\nScores com máscara:")
print(masked_scores)

attention = row_softmax(masked_scores)

print("\nAttention após softmax:")
print(attention)