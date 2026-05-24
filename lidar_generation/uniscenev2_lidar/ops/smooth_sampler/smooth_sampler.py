import torch
from torch import nn as nn
from torch.autograd import Function
from . import smooth_sampler_ext

bool_sample = smooth_sampler_ext.bool_sample

def padding_mode_enum(padding_mode):
    if padding_mode == "zeros":
        return 0
    elif padding_mode == "border":
        return 1
    else:  # padding_mode == 'reflection'
        return 2


class SmoothSamplerBackward(Function):
    @staticmethod
    def forward(ctx, input, grid, grad_out, padding_mode="zeros", align_corners=True, apply_smoothstep=False):
        ctx.align_corners = align_corners
        ctx.apply_smoothstep = apply_smoothstep
        ctx.padding_mode = padding_mode
        grad_input, grad_grid = smooth_sampler_ext.backward(grad_out, input, grid, padding_mode_enum(padding_mode),
                                            ctx.align_corners, apply_smoothstep, input.requires_grad)
        ctx.save_for_backward(input, grid, grad_out)
        
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad_out_input, grad_out_grid):
        input, grid, grad_out = ctx.saved_tensors

        input_requires_grad = grad_out_input is not None and (grad_out_input != 0.).any().item()
        grad_input, grad_grid, grad_grad_out = smooth_sampler_ext.backward_backward(grad_out_input.contiguous(), grad_out_grid.contiguous(),
                                                                    input, grid, grad_out,
                                                                    padding_mode_enum(ctx.padding_mode), ctx.align_corners,
                                                                    ctx.apply_smoothstep, input_requires_grad)
        
        return grad_input, grad_grid, grad_grad_out, None, None, None


class SmoothSampler(Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode="zeros", align_corners=True, apply_smoothstep=False):
        output = smooth_sampler_ext.forward(input, grid, padding_mode_enum(padding_mode), align_corners, apply_smoothstep)
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        ctx.apply_smoothstep = apply_smoothstep
        ctx.padding_mode = padding_mode
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, grid = ctx.saved_tensors
        
        if (grad_out == 0.).all().item():
            return torch.zeros_like(input), torch.zeros_like(grid), None, None, None
         
        d_input, d_grid = SmoothSamplerBackward.apply(input, grid, grad_out.contiguous(), ctx.padding_mode, ctx.align_corners, ctx.apply_smoothstep)
        return d_input, d_grid, None, None, None


if __name__ == "__main__":
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    
    for padding_mode in ["zeros", "border", "reflection"]:
        for align_corners in [True, False]:
            input = (torch.rand([2,2,2,3,11], device="cuda")).requires_grad_(True)
            grid = (torch.rand([2,2,1,5,3], device="cuda") * 2. - 1.).requires_grad_(True)
    
            # SmoothSampler forward vs native forward
            out1 = SmoothSampler.apply(input, grid, padding_mode, align_corners, False)
            out2 = torch.nn.functional.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
            assert torch.allclose(out1, out2)
    
            # SmoothSampler backward vs native backward
            grad1_input, grad1_grid = torch.autograd.grad(out1, [input, grid], torch.ones_like(out1), create_graph=True)
            grad2_input, grad2_grid = torch.autograd.grad(out2, [input, grid], torch.ones_like(out2), create_graph=True)
            assert torch.allclose(grad1_input, grad2_input)
            assert torch.allclose(grad1_grid, grad2_grid)
            
            for apply_smoothstep in [True, False]:
                input = (torch.rand([2,2,2,3,11], device="cuda").double()).requires_grad_(True)
                grid = (torch.rand([2,2,1,5,3], device="cuda") * 2. - 1.).double().requires_grad_(True)
                
                # Analytic gradients vs finite differences gradients
                torch.autograd.gradcheck(SmoothSampler.apply,
                                            [input, grid, padding_mode, align_corners, apply_smoothstep],
                                            eps=1e-4, atol=1e-3, rtol=1e-2)
                torch.autograd.gradgradcheck(SmoothSampler.apply,
                                                [input, grid, padding_mode, align_corners, apply_smoothstep],
                                                eps=1e-4, atol=1e-3, rtol=1e-2)