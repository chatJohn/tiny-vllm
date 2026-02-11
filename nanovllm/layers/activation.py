import torch
import torch.nn.functional as F
from torch import nn
from .cuda import project_cuda_ops

class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    # @torch.compile
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x, y = x.chunk(2, -1)
    #     return F.silu(x) * y
    # 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        project_cuda_ops.silu_and_mul_bf16(output, x, y)
        return  output