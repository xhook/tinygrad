import numpy as np

from tinygrad import Device, Tensor
from tinygrad.codegen.uops import UOpGraph, UOps, UOp
from tinygrad.device import CompiledASTRunner, Buffer
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.ops import BinaryOps


def create_graph(idx_len: int, emb_len: int):
  uops = []
  uops += [UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(0, 'out', True))]  # 0
  uops += [UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(1, 'dict', True))] # 1
  uops += [UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.int32), arg=(2, 'idx', True))]    # 2
  uops += [UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=0)]                                     # 3
  uops += [UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=idx_len)]                               # 4
  uops += [UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=emb_len)]                               # 5
  uops += [UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(uops[3], uops[4]))]                     # 6: loop over indices
  uops += [UOp(uop=UOps.LOAD, dtype=dtypes.int32, vin=(uops[2], uops[5]))]                     # 7: load the index
  uops += [UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(uops[7], uops[5]), arg=BinaryOps.MUL)]   # 8: dimension index start
  uops += [UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(uops[8], uops[5]), arg=BinaryOps.ADD)]   # 9: dimension index end
  uops += [UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(uops[8], uops[9]))]                     # 10: loop over embedding dimensions
  uops += [UOp(uop=UOps.LOAD, dtype=dtypes.float32, vin=(uops[1], uops[10]))]                  # 11: load an embedding at index
  # ... not finished, complete the loop over embedding dimensions
  uops += [UOp(uop=UOps.STORE, dtype=None, vin=(uops[0], uops[5], uops[7]))]                   # 12: store embedding to the output
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(uops[5],))]                                  # 13
  return UOpGraph(uops)


def _uops_to_prg(uops):
  device = Device[Device.DEFAULT]
  src = device.compiler.render("test", uops)
  print(src)
  has_local = device.compiler.linearizer_opts.has_local
  return CompiledASTRunner("test", src, device, [1] if has_local else None, [1] if has_local else None)


def embedding(dict_t, idx_t):
  dict_r = dict_t.realize()
  idx_r = idx_t.realize()
  out_r = Tensor.empty(*idx_t.shape, dtype=dtypes.float32).realize()
  g = create_graph(3, dict_t.shape[1])
  g.print()
  c_prg = _uops_to_prg(g)
  c_prg.exec([out_r.lazydata.realized, dict_r.lazydata.realized, idx_r.lazydata.realized])
  return out_r


def main():
  dict_t = Tensor([
    [1.2, 6.2, 9.1],
    [2.1, 77.7, 14.4],
    [-8.1, 22.11, 2.0],
    [-55.1, 2.5, 22.1],
    [0.1, -23, -10]
  ], dtype=dtypes.float32)
  idx_t = Tensor([1, 2, 4], dtype=dtypes.int32)
  out_r = embedding(dict_t, idx_t)
  print(out_r.numpy())
  assert np.allclose(out_r.numpy(), [[2.1, 77.7, 14.4], [-8.1, 22.11, 2.0], [0.1, -23, -10]])


if __name__ == "__main__":
  main()
