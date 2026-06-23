import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(out, inp, stride, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(inp + row*stride + cols, mask=cols<n, other=-float("inf"))
    tl.store(out + row*stride + cols, tl.softmax(x), mask=cols<n)
