#!/bin/bash
# Environment for USDNet on an RTX 5090 (sm_120).
# Usage:  source /data/challenge/USDNet/env.sh
#
# The repo targets CUDA 11.3 / torch 1.12 and its Dockerfile builds for
# sm_70..8.9 -- none of which run on a 5090. Instead the `usdnet` env is
# python 3.10 + torch 2.7.1+cu128 + MinkowskiEngine 0.5.8 (alpsaur/MinkowskiEngine@8f76f27)
# built for sm_120, plus USDNet's own deps.

export USDNET_ENV=$HOME/.conda/envs/usdnet

# nvcc 12.8.61 lives inside the conda env and matches torch's cu128. The system
# nvcc is 11.5 and /usr/local/cuda points at 13.2; either one makes torch's
# cpp_extension abort with a CUDA version mismatch.
export CUDA_HOME=$USDNET_ENV
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# Blackwell. Needed whenever a CUDA extension gets rebuilt.
export TORCH_CUDA_ARCH_LIST="12.0"

export OMP_NUM_THREADS=3          # MinkowskiEngine is faster with this low
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# conda's cuda-nvcc activation hook appends to this without defaulting it,
# which aborts any caller running under `set -u`.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

eval "$(conda shell.bash hook)" 2>/dev/null && conda activate usdnet
