#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def reduce(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
	pid = tl.program_id(0)
	block_start = pid * BLOCK_SIZE

	offsets = block_start + tl.arange(0, BLOCK_SIZE)
	mask = offsets < num_elements

	vals = tl.load(input_ptr + offsets, mask=mask, other=0)
	acc = tl.sum(vals)

	tl.atomic_add(output_ptr + 0, acc)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
	args = parser.parse_args()

	data_type = torch.int32
	BLOCK_SIZE = 128
	num_elements = 1_000_000
	grid = lambda META: (triton.cdiv(num_elements, META["BLOCK_SIZE"]),)

	x = torch.randint(0, 42, (num_elements,), dtype=data_type, device="cuda")
	y = torch.zeros(1, dtype=data_type, device="cuda")

	reduce[grid](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

	actual = y.item()

	if args.validate:
		expected = int(x.cpu().sum())
		print(f"Expected: {expected:,} Actual: {actual:,}")
		if actual != expected:
			print("Validation Failed: Triton output does not match PyTorch output.")
			sys.exit(1)
		else:
			print("Validation Successful!")
	else:
		print(f"Actual: {actual:,}")


if __name__ == "__main__":
	main()
