import numpy as np

from tinygrad import Device
from tinygrad.codegen.uops import UOpGraph, UOps, UOp
from tinygrad.device import CompiledASTRunner, Buffer
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage


def create_graph():
  uop0 = UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float), arg=(0, 'data0', True))
  uop1 = UOp(uop=UOps.CONST, dtype=dtypes.int, arg=0)
  uop2 = UOp(uop=UOps.CONST, dtype=dtypes.int, arg=3)
  uop3 = UOp(uop=UOps.CONST, dtype=dtypes.float, arg=0.0)
  uop4 = UOp(uop=UOps.LOOP, dtype=dtypes.int, vin=(uop1, uop2))
  uop5 = UOp(uop=UOps.STORE, dtype=None, vin=(uop0, uop4, uop3))
  uop6 = UOp(uop=UOps.ENDLOOP, dtype=None, vin=(uop4,))
  g = UOpGraph([uop0, uop1, uop2, uop3, uop4, uop5, uop6])
  return g


def _uops_to_prg(uops):
  src = Device[Device.DEFAULT].compiler.render("test", uops)
  print(src)
  has_local = Device[Device.DEFAULT].compiler.linearizer_opts.has_local
  return CompiledASTRunner("test", src, Device[Device.DEFAULT], [1] if has_local else None, [1] if has_local else None)


if __name__ == "__main__":
  g = create_graph()
  g.print()

  out_buf = Buffer(Device.DEFAULT, 3, dtypes.float)
  c_prg = _uops_to_prg(g)
  c_prg.exec([out_buf])
  ret = np.empty(3, dtypes.float.np)
  out_buf.copyout(ret.data)
  print(ret)
