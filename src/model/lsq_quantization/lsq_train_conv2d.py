import torch
import torch.nn as nn
import torch.nn.functional as F


class LSQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, s_grad_scale, Qn, Qp):
        ctx.save_for_backward(x, scale)
        ctx.s_grad_scale = s_grad_scale
        ctx.Qn = Qn
        ctx.Qp = Qp
        
        x_scaled = x / scale
        x_clamped = torch.clamp(torch.round(x_scaled), Qn, Qp)
        x_quant = x_clamped * scale
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        s_grad_scale = ctx.s_grad_scale
        Qn, Qp = ctx.Qn, ctx.Qp
        
        x_scale = x / scale
        indicate_small = (x_scale <= Qn).float()
        indicate_big = (x_scale >= Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_scale = (indicate_small * Qn + indicate_big * Qp + 
                  indicate_middle * (-x_scale + torch.round(x_scale))) * grad_output
        grad_scale = grad_scale.sum().unsqueeze(0) * s_grad_scale
        
        grad_x = indicate_middle * grad_output
        
        return grad_x, grad_scale, None, None, None


class QAConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, n_bits=8, static_padding=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.n_bits = n_bits
        self.Qn = -2**(n_bits - 1)
        self.Qp = 2**(n_bits - 1) - 1
        
        self.static_padding = static_padding 
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.x_scale = nn.Parameter(torch.ones(1))        
        self.initialized = False
        
        self.register_buffer('init_state', torch.tensor(False, dtype=torch.bool))
        with torch.no_grad():
            self.weight_scale.data.copy_(2 * self.weight.abs().mean() / (self.Qp**0.5))

    def forward(self, x):
        if self.static_padding is not None:
            x = self.static_padding(x)
        if self.training and not self.init_state:
            with torch.no_grad():
                self.x_scale.data.copy_(2 * max(x.abs().mean(), 1e-4) / (self.Qp**0.5))
                self.init_state.fill_(True)
        if self.training:
            weight_s_grad_scale = 1.0 / ((self.weight.numel() * self.Qp)**0.5)
            x_s_grad_scale = 1.0 / ((x.numel() * self.Qp)**0.5)
        else:
            weight_s_grad_scale =  x_s_grad_scale = 1.0

        quantized_weight = LSQFunction.apply(self.weight, self.weight_scale, weight_s_grad_scale, self.Qn, self.Qp)
        quantized_x = LSQFunction.apply(x, self.x_scale, x_s_grad_scale, self.Qn, self.Qp)

        return F.conv2d(quantized_x, quantized_weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)
