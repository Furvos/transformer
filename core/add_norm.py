from transformer.encoder.layer_norm.layer_norm import layer_norm

def add_norm(x, sublayer_output, gamma, beta):
    return layer_norm(x + sublayer_output, gamma, beta)