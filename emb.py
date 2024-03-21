import numpy as np
from tinygrad.nn import Embedding
from tinygrad import Tensor, dtypes


def main():
  emb = Embedding(50, 8)
  idx = Tensor([
    [1, 2, 4],
    [7, 8, 22]
  ], dtype=dtypes.int32)
  out = emb(idx)
  print(out.numpy())


if __name__ == "__main__":
  main()
