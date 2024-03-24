from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.uops import UOpGraph, UOps, UOp
from tinygrad.device import Buffer, CompiledASTRunner
from tinygrad.dtype import PtrDType
from tinygrad.lazy import create_lazybuffer, LazyBuffer
from tinygrad.ops import BinaryOps, LoadOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.tensor import Function
from tinygrad.nn import Embedding


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
    srcs=(voc, idx)
  )


class EmbeddingFn(Function):
  def forward(self, voc: LazyBuffer, idx: LazyBuffer) -> LazyBuffer:
    return embedding_fwd(voc, idx)

  def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    raise NotImplementedError("backward not implemented for EmbeddingFn")


class EmbeddingV2:
  def __init__(self, vocab_size: int, embed_size: int):
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self.weight = Tensor.glorot_uniform(vocab_size, embed_size)

  def __call__(self, idx: Tensor) -> Tensor:
    return EmbeddingFn.apply(self.weight, idx)


def main():
  voc = Tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
  ], dtype=dtypes.float32)
  idx = Tensor([[1, 2, 4], [3, 3, 1]])
  emb_new = EmbeddingV2(5, 2)
  emb_new.weight = voc
  out = emb_new(idx)
  print(out.numpy())

  emb = Embedding(5, 2)
  emb.weight = voc

  orig_out = emb(idx)
  print(orig_out.numpy())

  assert (out.numpy() == orig_out.numpy()).all()


if __name__ == "__main__":
  main()
