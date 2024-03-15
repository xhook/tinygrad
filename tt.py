import tinygrad as tg

# t = tg.Tensor.arange(0, 30)
# t = tg.Tensor.zeros(3)
t0 = tg.Tensor([3.14159265])
t = tg.Tensor.sin(t0)
# idx = tg.Tensor.arange(300, 800, dtype=tg.dtypes.int32)
# print(t[idx].numpy())
print(t.numpy())
