import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from plane_fitting_c import plane_fitting_foward


class PlaneFittingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor, #[N,C=1,H,W]
        iter: int,
        sigma: float,
        kernel_size: int,
        min_disp: float,
        max_disp: float,
    ):
        assert input.is_cuda
        assert len(input.size()) == 4

        unfold = F.unfold(
            input,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        ) # [N, k*k, L], L=H*W
        
        # [N, k, k, L]
        unfold = unfold.reshape(-1, kernel_size, kernel_size, unfold.size(2))
        
        # [N, 2, L], here 2 for d_x, and d_y
        unfold = plane_fitting_foward(unfold, iter, sigma, min_disp, max_disp)
        
        fold = F.fold(unfold, 
                    output_size = input.size()[2:], 
                    kernel_size = 1) #[N,2,H,W]
        return fold


plane_fitting = PlaneFittingFunction.apply
