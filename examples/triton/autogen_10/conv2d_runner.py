#!/usr/bin/env python3
import torch
import triton
import argparse
from conv2d import conv2d


def main(validate=False):
    # Setup a small test case
    batch_size = 32
    channels = 12
    height = 128
    width = 128
    kernels = 16
    kernel_height = 64
    kernel_width = 64

    # Move to GPU and set dtype
    device = 'cuda:0'
    dtype = torch.float32

    # Create random input, kernel, and bias tensors
    input_tensor = torch.randint(0, 10, (batch_size, channels, height, width), device=device, dtype=dtype)
    kernel_tensor = torch.randint(0, 10, (kernels, channels, kernel_height, kernel_width), device=device, dtype=dtype)
    bias_tensor = torch.randn(kernels, device=device, dtype=dtype)

    # Warm-up
    for _ in range(10):
        y_triton = conv2d(input_tensor, kernel_tensor, bias_tensor)
    torch.cuda.synchronize()

    # Benchmark
    rep = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        y_triton = conv2d(input_tensor, kernel_tensor, bias_tensor)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    runtime_ms = elapsed_time_ms / rep
    print(f"Triton kernel runtime: {runtime_ms:.4f} ms")

    if validate:
        # PyTorch's Conv2d for comparison
        conv_layer = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=kernels,
            kernel_size=(kernel_height, kernel_width),
            stride=(kernel_height, kernel_width),
            bias=True,
            dtype=dtype
        ).to(device)

        # Copy kernel and bias to the torch layer
        with torch.no_grad():
            conv_layer.weight.copy_(kernel_tensor)
            conv_layer.bias.copy_(bias_tensor)

        y_torch = conv_layer(input_tensor)

        # Validate the results
        if torch.allclose(y_torch, y_triton, atol=1e-2, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print("Triton output:\n", y_triton)
            print("PyTorch output:\n", y_torch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Conv2D Benchmark")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(validate=args.validate) 