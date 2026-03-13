import numpy as np
from transformer.decoder.cross_attention.cross_attention import cross_attention

batch_size = 1
seq_len_frances = 10
seq_len_ingles = 4
d_model = 512

encoder_output = np.random.randn(batch_size, seq_len_frances, d_model)

decoder_state = np.random.randn(batch_size, seq_len_ingles, d_model)

output, weights = cross_attention(encoder_output, decoder_state)

print("Shape encoder output:")
print(encoder_output.shape)

print("\nShape decoder state:")
print(decoder_state.shape)

print("\nCross attention output:")
print(output.shape)

print("\nAttention weights:")
print(weights.shape)