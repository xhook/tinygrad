import numpy as np
from tinygrad.nn import Embedding, EmbeddingV2
from tinygrad import Tensor

def main():
  s = 1
  B, T, embed_size, vocab_size = 4 * s, 3 * s, 2 * s, 5 * s
  x = Tensor(np.random.randint(0, vocab_size, (B, T)))
  layer_old = Embedding(vocab_size, embed_size)
  layer_new = EmbeddingV2(vocab_size, embed_size)
  layer_new.weight = layer_old.weight

  y_old = layer_old(x).numpy()
  print("old:", y_old)
  y_new = layer_new(x).numpy()
  print("new:", y_new)

if __name__ == "__main__":
  main()