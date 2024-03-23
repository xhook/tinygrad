import numpy as np
from tinygrad import dtypes, Device, Tensor
from tinygrad.device import Buffer, CompiledASTRunner
from tinygrad.lazy import create_lazybuffer, LazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.shape.shapetracker import ShapeTracker

from emb_uop import create_graph, _uops_to_prg

def embedding(dict_t: Tensor, idx_t: Tensor) -> Tensor:
  idx_len = idx_t.shape[0]
  emb_len = dict_t.shape[1]

  def emb(out: Buffer, voc: Buffer, idx: Buffer):
    graph = create_graph(idx_len=idx_len, emb_len=emb_len)
    return _uops_to_prg(graph, name="emb").exec([out, voc, idx])

  return Tensor(create_lazybuffer(
    dict_t.device,
    ShapeTracker.from_shape((idx_len, emb_len)),
    dtypes.float32,
    LoadOps.CUSTOM,
    arg=emb,
    srcs=(dict_t.lazydata, idx_t.lazydata)
  ))


def main():
  dict_t = Tensor([
    [1.2, 6.2],
    [2.1, 77.7],
    [-8.1, 22.11],
    [-55.1, 2.5],
    [0.1, -23]
  ], dtype=dtypes.float32)
  idx_t = Tensor([1, 2, 4], dtype=dtypes.int32)
  out = embedding(dict_t, idx_t)
  print(out.numpy())
  assert np.allclose(out.numpy(), [[2.1, 77.7], [-8.1, 22.11], [0.1, -23]])


if __name__ == "__main__":
  main()
