#!/bin/bash

echo "Running gemm_runner.py..."
python3 gemm_runner.py --version swizzle --m 1024 --n 1024 --k 1024
python3 gemm_runner.py --version unswizzle --m 1024 --n 1024 --k 1024

echo "Running layer_norm_runner.py..."
python3 layer_norm_runner.py --version swizzle --batch_size 64 --seq_len 1024 --hidden_size 1024
python3 layer_norm_runner.py --version unswizzle --batch_size 64 --seq_len 1024 --hidden_size 1024

echo "Running softmax_runner.py..."
python3 softmax_runner.py --version swizzle --batch_size 64 --seq_len 1024 --vocab_size 50257
python3 softmax_runner.py --version unswizzle --batch_size 64 --seq_len 1024 --vocab_size 50257

echo "Running smith_waterman_runner.py..."
python3 smith_waterman_runner.py --version swizzle --seq1_len 1024 --seq2_len 1024
python3 smith_waterman_runner.py --version unswizzle --seq1_len 1024 --seq2_len 1024

echo "Running stencil_2d_runner.py..."
python3 stencil_2d_runner.py --version swizzle --grid_size 1024
python3 stencil_2d_runner.py --version unswizzle --grid_size 1024

echo "All runners executed." 