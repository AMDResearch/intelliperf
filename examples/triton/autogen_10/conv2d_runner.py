import torch
import triton
import argparse
from conv2d import conv2d


def main(validate=False):
    # Setup a small test case
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    kernels = 4
    kernel_height = 4
    kernel_width = 4

    # Move to GPU and set dtype
    device = 'cuda:0'
    dtype = torch.float32

    # Create random input, kernel, and bias tensors
    input_tensor = torch.randint(0, 10, (batch_size, channels, height, width), device=device, dtype=dtype)
    kernel_tensor = torch.randint(0, 10, (kernels, channels, kernel_height, kernel_width), device=device, dtype=dtype)
    bias_tensor = torch.randn(kernels, device=device, dtype=dtype)

    # Run Triton implementation
    y_triton = conv2d(input_tensor, kernel_tensor, bias_tensor)

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