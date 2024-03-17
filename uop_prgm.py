import numpy as np

from tinygrad import Device
from tinygrad.codegen.uops import UOpGraph, UOps, UOp
from tinygrad.device import CompiledASTRunner, Buffer
from tinygrad.dtype import PtrDType, dtypes


def create_graph(idx_len: int):
  uops = []
  uops += [UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(0, 'out', True))]  # 0
  uops += [UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(1, 'dict', True))] # 1
  uops += [UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.int32), arg=(2, 'idx', True))]    # 2
  uops += [UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=0)]                                     # 3
  uops += [UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=idx_len)]                               # 4
  uops += [UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(uops[3], uops[4]))]                     # 5
  uops += [UOp(uop=UOps.LOAD, dtype=dtypes.int32, vin=(uops[2], uops[5]))]                     # 6
  uops += [UOp(uop=UOps.LOAD, dtype=dtypes.float32, vin=(uops[1], uops[6]))]                   # 7
  uops += [UOp(uop=UOps.STORE, dtype=None, vin=(uops[0], uops[5], uops[7]))]                   # 8
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(uops[5],))]                                  # 9
  return UOpGraph(uops)


def _uops_to_prg(uops):
  device = Device[Device.DEFAULT]
  src = device.compiler.render("test", uops)
  print(src)
  has_local = device.compiler.linearizer_opts.has_local
  return CompiledASTRunner("test", src, device, [1] if has_local else None, [1] if has_local else None)


def main():
  out_buf = Buffer(Device.DEFAULT, 3, dtypes.float32)
  dict_buf = Buffer(Device.DEFAULT, 5, dtypes.float32)
  dict_buf.copyin(np.array([1.2, 2.1, 77.7, 14.4, -8.1], dtype=dtypes.float32.np).data)
  idx_buf = Buffer(Device.DEFAULT, 3, dtypes.int32)
  idx_buf.copyin(np.array([1, 2, 4], dtype=dtypes.int32.np).data)
  g = create_graph(3)
  g.print()
  c_prg = _uops_to_prg(g)
  c_prg.exec([out_buf, dict_buf, idx_buf])
  ret = np.empty(3, dtypes.float32.np)
  out_buf.copyout(ret.data)
  print(ret)
  assert np.allclose(ret, [2.1, 77.7, -8.1])


if __name__ == "__main__":
  main()
