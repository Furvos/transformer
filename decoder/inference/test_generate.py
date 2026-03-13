import numpy as np
from transformer.decoder.inference.generate import generate_next_token

VOCAB_SIZE = 10000

vocab = [f"token_{i}" for i in range(VOCAB_SIZE)]

START = "<START>"
EOS = "<EOS>"

vocab[0] = EOS

encoder_out = np.random.randn(1, 10, 512)

current_sequence = [START, "O", "gato"]

while True:

    probs = generate_next_token(current_sequence, encoder_out)

    next_token_id = np.argmax(probs)

    next_token = vocab[next_token_id]

    if next_token == EOS:
        print("\nFim da geração")
        break

    current_sequence.append(next_token)

    print("Sequência:", current_sequence)

    if len(current_sequence) > 15:
        break