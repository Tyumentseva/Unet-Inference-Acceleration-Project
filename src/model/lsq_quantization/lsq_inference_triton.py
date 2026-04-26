import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def lsq_conv_kernel_g1(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    batch, in_c, in_h, in_w, out_c, out_h, out_w,
    kh, kw, stride_h, stride_w, padding_h, padding_w,
    x_scale_ptr, combined_scale_ptr, qn_ptr, qp_ptr,
    M, N, K,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_w_out, stride_w_in, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    stride_bias,
    OUTPUT_TYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    batch_id = offs_m // (out_h * out_w)
    pix_id = offs_m % (out_h * out_w)
    out_y = pix_id // out_w
    out_x = pix_id % out_w

    x_scale = tl.load(x_scale_ptr)
    combined_scale = tl.load(combined_scale_ptr)
    qn = tl.load(qn_ptr)
    qp = tl.load(qp_ptr)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k_base in range(0, K, BLOCK_SIZE_K):
        offs_k = k_base + tl.arange(0, BLOCK_SIZE_K)
        k_ch = offs_k // (kh * kw)
        k_rem = offs_k % (kh * kw)
        k_h = k_rem // kw
        k_w = k_rem % kw

        cy = out_y[:, None] * stride_h + k_h[None, :] - padding_h
        cx = out_x[:, None] * stride_w + k_w[None, :] - padding_w
        
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K) & \
                 (cy >= 0) & (cy < in_h) & (cx >= 0) & (cx < in_w)
        off_x = (batch_id[:, None].to(tl.int64) * stride_xb + k_ch[None, :].to(tl.int64) * stride_xc + 
                 cy.to(tl.int64) * stride_xh + cx.to(tl.int64) * stride_xw)
        off_w = (offs_n[None, :].to(tl.int64) * stride_w_out + k_ch[:, None].to(tl.int64) * stride_w_in + 
                 k_h[:, None].to(tl.int64) * stride_w_kh + k_w[:, None].to(tl.int64) * stride_w_kw)
        
        a = tl.load(x_ptr + off_x, mask=mask_x, other=0.0)
        a = tl.math.round(a / x_scale)
        a = tl.maximum(a, qn)
        a = tl.minimum(a, qp)
        a = a.to(tl.int8)
        b = tl.load(w_ptr + off_w, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0).to(tl.int8)
        accumulator += tl.dot(a, b)

    bias_vals = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
    out = accumulator.to(OUTPUT_TYPE) * combined_scale + bias_vals[None, :].to(OUTPUT_TYPE)
    
    off_out = (batch_id[:, None].to(tl.int64) * stride_out_b + offs_n[None, :].to(tl.int64) * stride_out_c + 
               out_y[:, None].to(tl.int64) * stride_out_h + out_x[:, None].to(tl.int64) * stride_out_w)
    tl.store(out_ptr + off_out, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class LSQConv2dTriton(nn.Module):
    def __init__(self, lsq_layer):
        super().__init__()
        self.stride = lsq_layer.stride
        self.padding = lsq_layer.padding
        self.static_padding = getattr(lsq_layer, 'static_padding', None)
        
        device = lsq_layer.weight.device

        self.register_buffer('Qn', torch.tensor(lsq_layer.Qn, dtype=torch.float32, device=device))
        self.register_buffer('Qp', torch.tensor(lsq_layer.Qp, dtype=torch.float32, device=device))
        self.register_buffer('x_scale', lsq_layer.x_scale.data.clone().to(device))
        
        with torch.no_grad():
            sw = lsq_layer.weight_scale.data
            sa = lsq_layer.x_scale.data
            self.register_buffer('combined_scale', (sw * sa).clone().to(device))

            w_int = torch.clamp(torch.round(lsq_layer.weight.data / sw), lsq_layer.Qn, lsq_layer.Qp).to(torch.int8)
            self.register_buffer('weight_int', w_int.to(memory_format=torch.channels_last))

            if lsq_layer.bias is not None:
                self.register_buffer('bias', lsq_layer.bias.data.clone())
            else:
                self.register_buffer('bias', torch.zeros(lsq_layer.out_channels, device=sw.device))
                
        self.out_c, self.in_c, self.kh, self.kw = self.weight_int.shape
        self.K_val = self.in_c * self.kh * self.kw
        self.sh, self.sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        self.ph, self.pw = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding

    def _apply(self, fn):
        super()._apply(fn)
        self.weight_int = self.weight_int.to(torch.int8)
        return self

    def forward(self, x):
        target_dtype = x.dtype
 
        triton_output_type = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16
        }[target_dtype]

        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)

        if self.static_padding is not None:
            x = self.static_padding(x)
        
        B, C, H, W = x.shape
        OH, OW = (H + 2*self.ph - self.kh) // self.sh + 1, (W + 2*self.pw - self.kw) // self.sw + 1
        M = B * OH * OW
        
        output = torch.empty((B, self.out_c, OH, OW), device=x.device, 
                             dtype=target_dtype, memory_format=torch.channels_last)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(self.out_c, META['BLOCK_SIZE_N']),)

        lsq_conv_kernel_g1[grid](
            x, self.weight_int, self.bias, output,
            B, C, H, W, self.out_c, OH, OW, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw,
            self.x_scale, self.combined_scale, self.Qn, self.Qp,
            M, self.out_c, self.K_val,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight_int.stride(0), self.weight_int.stride(1), self.weight_int.stride(2), self.weight_int.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            self.bias.stride(0), OUTPUT_TYPE=triton_output_type,
        )
        return output
