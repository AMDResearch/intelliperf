
import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C, H, W,
    F, R, S,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_F: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid_n = tl.program_id(axis=0)
    pid_f = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_w = tl.program_id(axis=3)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_f = pid_f * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    offs_y_n = offs_n[:, None, None, None]
    offs_y_f = offs_f[None, :, None, None]
    offs_y_h = offs_h[None, None, :, None]
    offs_y_w = offs_w[None, None, None, :]

    y_ptrs = y_ptr + offs_y_n * stride_yn + offs_y_f * stride_yc + \
             offs_y_h * stride_yh + offs_y_w * stride_yw

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_F, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for c in range(0, C, BLOCK_SIZE_C):
        for r in range(0, R, BLOCK_SIZE_R):
            for s in range(0, S, BLOCK_SIZE_S):
                offs_c = c + tl.arange(0, BLOCK_SIZE_C)
                offs_r = r + tl.arange(0, BLOCK_SIZE_R)
                offs_s = s + tl.arange(0, BLOCK_SIZE_S)
                
                offs_x_n = offs_n[:, None, None, None, None, None]
                offs_x_c = offs_c[None, :, None, None, None, None]
                offs_x_h = (offs_h[None, None, :, None, None, None] + offs_r[None, None, None, None, :, None])
                offs_x_w = (offs_w[None, None, None, :, None, None] + offs_s[None, None, None, None, None, :])

                x_ptrs = x_ptr + offs_x_n * stride_xn + offs_x_c * stride_xc + \
                         offs_x_h * stride_xh + offs_x_w * stride_xw

                mask_x = (offs_x_n < N) & (offs_x_c < C) & (offs_x_h < H) & (offs_x_w < W)
                x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

                offs_w_f = offs_f[None, :, None, None, None, None]
                offs_w_c = offs_c[:, None, None, None, None, None]
                offs_w_r = offs_r[None, None, :, None, None, None]
                offs_w_s = offs_s[None, None, None, :, None, None]

                w_ptrs = w_ptr + offs_w_f * stride_wn + offs_w_c * stride_wc + \
                         offs_w_r * stride_wh + offs_w_s * stride_ww
                
                mask_w = (offs_w_f < F) & (offs_w_c < C) & (offs_w_r < R) & (offs_w_s < S)
                w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
                
                accumulator += tl.sum(x_tile[:, :, :, :, :, :] * w_tile[:, :, :, :, :, :], axis=(1, 4, 5))

    y = accumulator.to(tl.float16)
    mask_y = (offs_y_n < N) & (offs_y_f < F) & (offs_y_h < H) & (offs_y_w < W)
    tl.store(y_ptrs, y, mask=mask_y)


def conv2d(x, w):
    N, C, H, W = x.shape
    F, _, R, S = w.shape
    P_H, P_W = H - R + 1, W - S + 1 
    y = torch.empty((N, F, P_H, P_W), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']), 
        triton.cdiv(F, META['BLOCK_SIZE_F']),
        triton.cdiv(P_H, META['BLOCK_SIZE_H']),
        triton.cdiv(P_W, META['BLOCK_SIZE_W'])
    )

    conv2d_kernel[grid](
        x, w, y,
        N, C, H, W,
        F, R, S,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_SIZE_N=16, BLOCK_SIZE_F=32,
        BLOCK_SIZE_H=32, BLOCK_SIZE_W=32,
        BLOCK_SIZE_C=16, BLOCK_SIZE_R=3, BLOCK_SIZE_S=3
    )
    return y

def main():
    N, C, H, W = 128, 256, 128, 128
    F, R, S = 512, 3, 3

    x = torch.randn((N, C, H, W), device='cuda', dtype=torch.float16)
    w = torch.randn((F, C, R, S), device='cuda', dtype=torch.float16)
    
    rep = 100
    
    for _ in range(10):
        y_triton = conv2d(x, w)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        y_triton = conv2d(x, w)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton conv2d time: {triton_time:.4f} ms")

    # y_torch = torch.nn.functional.conv2d(x, w)
    # print(f"Triton output: {y_triton}")
    # print(f"Torch output: {y_torch}")
    # assert torch.allclose(y_triton, y_torch, atol=1e-1, rtol=0), "Triton and PyTorch results differ"


if __name__ == "__main__":
    main() 