import numpy as np
from tinygrad import dtypes, Device, Tensor
from tinygrad.device import Buffer, CompiledASTRunner
from tinygrad.lazy import create_lazybuffer, LazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.shape.shapetracker import ShapeTracker


def emb_gpu(out: Buffer, voc: Buffer, idx: Buffer):
  src = """
extern "C" __global__ void emb_gpu(float* out, float* dict, int* idx) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    int val0 = idx[ridx0];
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      float val1 = dict[(val0*2)+ridx1];
      out[(ridx0*2)+ridx1] = val1;
    }
  }
}
  """
  device = Device[Device.DEFAULT]
  has_local = device.compiler.linearizer_opts.has_local
  return CompiledASTRunner("emb_gpu", src, device, [1] if has_local else None, [1] if has_local else None).exec([out, voc, idx])


def emb_cpu(out: Buffer, voc: Buffer, idx: Buffer):
  src = """
    void emb_cpu(float* out, float* dict, int* idx) {
      for (int ridx0 = 0; ridx0 < 3; ridx0++) {
        int val0 = idx[ridx0];
        for (int ridx1 = 0; ridx1 < 2; ridx1++) {
          float val1 = dict[(val0*2)+ridx1];
          out[(ridx0*2)+ridx1] = val1;
        }
      }
    }
  """
  device = Device[Device.DEFAULT]
  has_local = device.compiler.linearizer_opts.has_local
  return CompiledASTRunner("emb_cpu", src, device, [1] if has_local else None, [1] if has_local else None).exec([out, voc, idx])


def embedding(dict_t: Tensor, idx_t: Tensor) -> Tensor:
  return Tensor(create_lazybuffer(
    dict_t.device,
    ShapeTracker.from_shape((idx_t.shape[0], dict_t.shape[1])),
    dtypes.float32,
    LoadOps.CUSTOM,
    arg=emb_gpu,
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
