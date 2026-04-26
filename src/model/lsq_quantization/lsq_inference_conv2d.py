import torch
import torch.nn as nn
import torch.nn.functional as F


class QAConv2dInference(nn.Conv2d):
    def __init__(self, original_layer):
        device = original_layer.weight.device
        super().__init__(
            original_layer.in_channels, original_layer.out_channels, original_layer.kernel_size,
            stride=original_layer.stride, padding=original_layer.padding,
            dilation=original_layer.dilation, groups=original_layer.groups,
            bias=(original_layer.bias is not None)
        )
        self.to(device)
        self.static_padding = original_layer.static_padding

        self.register_buffer('n_bits', torch.tensor(original_layer.n_bits, dtype=torch.int32, device=device))
        self.register_buffer('Qn', torch.tensor(original_layer.Qn, dtype=torch.float32, device=device))
        self.register_buffer('Qp', torch.tensor(original_layer.Qp, dtype=torch.float32, device=device))        
        self.register_buffer('x_scale', original_layer.x_scale.data.clone().to(device))

        with torch.no_grad():
            s_w = original_layer.weight_scale.clamp(min=1e-8)
            w_q = torch.clamp(torch.round(original_layer.weight / s_w), self.Qn, self.Qp) * s_w
            self.weight.copy_(w_q)
            if original_layer.bias is not None:
                self.bias.copy_(original_layer.bias)

    def forward(self, x):
        if self.static_padding is not None:
            x = self.static_padding(x)
        s_x = self.x_scale.clamp(min=1e-8)
        x = torch.clamp(torch.round(x / s_x), self.Qn, self.Qp) * s_x
        return super().forward(x)
