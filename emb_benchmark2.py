import numpy as np
from tqdm import trange

from tinygrad import Tensor
from tinygrad.nn import Embedding
from tinygrad.nn import EmbeddingV2


def main():
  s = 1
  B, T, embed_size, vocab_size = 128 * s, 32 * s, 32 * s, 2048 * s
  x = Tensor(np.random.randint(0, vocab_size, (B, T)))
  layer_old = Embedding(vocab_size, embed_size)
  for _ in trange(10):
    layer_old(x).realize()

  # layer_new = EmbeddingV2(vocab_size, embed_size)
  # for _ in trange(10):
  #   layer_new(x).realize()


if __name__ == "__main__":
  main()
