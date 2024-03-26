from typing import Tuple, Optional

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.uops import UOpGraph, UOps, UOp
from tinygrad.device import Buffer, CompiledASTRunner
from tinygrad.dtype import PtrDType
from tinygrad.lazy import create_lazybuffer, LazyBuffer
from tinygrad.ops import BinaryOps, LoadOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.tensor import Function


def _uops_to_prg(uops, name="test"):
  device = Device[Device.DEFAULT]
  src = device.compiler.render(name, uops)
  has_local = device.compiler.linearizer_opts.has_local  # TODO: Look why this is needed, I simply copied it from test_custom_function.py
  return CompiledASTRunner(name, src, device, [1] if has_local else None, [1] if has_local else None)


def emb_fwd_graph(batch_size: int, idx_len: int, emb_len: int):
  uops = []
  uops += [p_out := UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(0, 'out', True))]
  uops += [p_voc := UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(1, 'voc', True))]
  uops += [p_idx := UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.int32), arg=(2, 'idx', True))]
  uops += [c_zero := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=0)]
  uops += [c_idx_len := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=idx_len)]
  uops += [c_emb_len := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=emb_len)]
  uops += [c_bs := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=batch_size)]
  uops += [l_bat_i := UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(c_zero, c_bs))]
  uops += [v_idx_i_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(l_bat_i, c_idx_len), arg=BinaryOps.MUL)]
  uops += [l_idx_i := UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(c_zero, c_idx_len))]
  uops += [v_idx_i := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_idx_i_st, l_idx_i), arg=BinaryOps.ADD)]
  uops += [v_idx := UOp(uop=UOps.LOAD, dtype=dtypes.int32, vin=(p_idx, v_idx_i))]
  uops += [l_emb_i := UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(c_zero, c_emb_len))]
  uops += [v_emb_idx_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_idx, c_emb_len), arg=BinaryOps.MUL)]
  uops += [v_emb_idx := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_emb_idx_st, l_emb_i), arg=BinaryOps.ADD)]
  uops += [v_emb := UOp(uop=UOps.LOAD, dtype=dtypes.float32, vin=(p_voc, v_emb_idx))]
  uops += [v_out_i1_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(l_bat_i, c_idx_len), arg=BinaryOps.MUL)]
  uops += [v_out_i1_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_out_i1_st, c_emb_len), arg=BinaryOps.MUL)]
  uops += [v_out_i2_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(l_idx_i, c_emb_len), arg=BinaryOps.MUL)]
  uops += [v_out_i1 := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_out_i1_st, v_out_i2_st), arg=BinaryOps.ADD)]
  uops += [v_out_i2 := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_out_i1, l_emb_i), arg=BinaryOps.ADD)]
  uops += [UOp(uop=UOps.STORE, dtype=None, vin=(p_out, v_out_i2, v_emb))]
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(l_emb_i,))]
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(l_idx_i,))]
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(l_bat_i,))]
  return UOpGraph(uops)


def embedding_fwd(voc: LazyBuffer, idx: LazyBuffer) -> LazyBuffer:
  bs, idx_len = idx.shape
  emb_len = voc.shape[1]
  graph = emb_fwd_graph(idx_len=idx_len, emb_len=emb_len, batch_size=bs)
  prg = _uops_to_prg(graph, name=f"emb_fwd_b{bs}_i{idx_len}_e{emb_len}")

  def ex(out: Buffer, voc: Buffer, idx: Buffer):
    return prg.exec([out, voc, idx])

  return create_lazybuffer(
    voc.device,
    ShapeTracker.from_shape((bs, idx_len, emb_len)),
    dtypes.float32,
    LoadOps.CUSTOM,
    arg=ex,
    srcs=(voc, idx.contiguous())
  )


def emb_bwd_graph(batch_size: int, idx_len: int, emb_len: int):
  uops = []
  uops += [p_voc_grad := UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(0, 'voc_grad', True))]
  uops += [p_idx := UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.int32), arg=(1, 'idx', True))]
  uops += [p_grad := UOp(uop=UOps.DEFINE_GLOBAL, dtype=PtrDType(dtypes.float32), arg=(2, 'grad', True))]
  uops += [c_zero := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=0)]
  uops += [c_idx_len := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=idx_len)]
  uops += [c_emb_len := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=emb_len)]
  uops += [c_bs := UOp(uop=UOps.CONST, dtype=dtypes.int32, arg=batch_size)]
  uops += [l_b := UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(c_zero, c_bs))]
  uops += [l_i := UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(c_zero, c_idx_len))]
  uops += [l_e := UOp(uop=UOps.LOOP, dtype=dtypes.int32, vin=(c_zero, c_emb_len))]
  uops += [v_idx_v_i_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(l_b, c_idx_len), arg=BinaryOps.MUL)]
  uops += [v_idx_v_i := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_idx_v_i_st, l_i), arg=BinaryOps.ADD)]
  uops += [v_idx_v := UOp(uop=UOps.LOAD, dtype=dtypes.int32, vin=(p_idx, v_idx_v_i))]
  uops += [v_grad_v_i_st1 := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(l_b, c_idx_len), arg=BinaryOps.MUL)]
  uops += [v_grad_v_i_st1 := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_grad_v_i_st1, c_emb_len), arg=BinaryOps.MUL)]
  uops += [v_grad_v_i_st2 := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(l_i, c_emb_len), arg=BinaryOps.MUL)]
  uops += [v_grad_v_i := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_grad_v_i_st1, v_grad_v_i_st2), arg=BinaryOps.ADD)]
  uops += [v_grad_v_i := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_grad_v_i, l_e), arg=BinaryOps.ADD)]
  uops += [v_grad_v := UOp(uop=UOps.LOAD, dtype=dtypes.float32, vin=(p_grad, v_grad_v_i))]
  uops += [v_voc_grad_v_i_st := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_idx_v, c_emb_len), arg=BinaryOps.MUL)]
  uops += [v_voc_grad_v_i := UOp(uop=UOps.ALU, dtype=dtypes.int32, vin=(v_voc_grad_v_i_st, l_e), arg=BinaryOps.ADD)]
  uops += [v_voc_grad_v_old := UOp(uop=UOps.LOAD, dtype=dtypes.float32, vin=(p_voc_grad, v_voc_grad_v_i))]
  uops += [v_voc_grad_v_new := UOp(uop=UOps.ALU, dtype=dtypes.float32, vin=(v_voc_grad_v_old, v_grad_v), arg=BinaryOps.ADD)]
  uops += [UOp(uop=UOps.STORE, dtype=None, vin=(p_voc_grad, v_voc_grad_v_i, v_voc_grad_v_new))]
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(l_e,))]
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(l_i,))]
  uops += [UOp(uop=UOps.ENDLOOP, dtype=None, vin=(l_b,))]
  return UOpGraph(uops)


def embedding_bwd(idx: LazyBuffer, grad: LazyBuffer, voc_len: int) -> LazyBuffer:
  bs, idx_len = idx.shape
  emb_len = grad.shape[2]
  graph = emb_bwd_graph(idx_len=idx_len, emb_len=emb_len, batch_size=bs)
  prg = _uops_to_prg(graph, name=f"emb_bwd_b{bs}_i{idx_len}_e{emb_len}")

  def ex(voc_grad: Buffer, idx: Buffer, grad: Buffer):
    return prg.exec([voc_grad, idx, grad])

  return create_lazybuffer(
    grad.device,
    ShapeTracker.from_shape((voc_len, emb_len)),
    dtypes.float32,
    LoadOps.CUSTOM,
    arg=ex,
    srcs=(idx.contiguous(), grad.contiguous())
  )


class EmbeddingFn(Function):
  def forward(self, voc: LazyBuffer, idx: LazyBuffer) -> LazyBuffer:
    self.idx = idx
    self.voc_shape = voc.shape
    return embedding_fwd(voc, idx)

  def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], None]:
    return embedding_bwd(self.idx, grad_output, self.voc_shape[0]), None


class Embedding:
  def __init__(self, vocab_size: int, embed_size: int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = Tensor.glorot_uniform(vocab_size, embed_size)

  def __call__(self, idx: Tensor) -> Tensor:
    idx32 = idx.cast(dtypes.int32)
    return EmbeddingFn.apply(self.weight, idx32)
