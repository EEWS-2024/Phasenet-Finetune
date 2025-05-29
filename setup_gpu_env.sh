#!/bin/bash
# Setup GPU Environment untuk PhaseNet Indonesia Training
# Jalankan: source setup_gpu_env.sh

echo "🔧 Setting up GPU environment for PhaseNet Indonesia..."

# CUDA Environment Variables
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/mooc_parallel_2021_003/miniconda3/envs/phasenet/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/mooc_parallel_2021_003/miniconda3/envs/phasenet/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/mooc_parallel_2021_003/miniconda3/envs/phasenet/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

# CUDA NVCC Path untuk libdevice (fix TensorFlow 2.15.0 XLA compilation)
export CUDA_NVCC_PATH="/home/mooc_parallel_2021_003/miniconda3/envs/phasenet/lib/python3.10/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_NVCC_PATH/bin:$PATH"

# TensorFlow GPU Optimizations
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Disable XLA JIT compilation to avoid libdevice issues (temporary workaround)
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"

echo "✅ GPU Environment setup complete!"
echo "📋 Current setup:"
echo "   CUDA_HOME: $CUDA_HOME"
echo "   CUDA_NVCC_PATH: $CUDA_NVCC_PATH"
echo "   TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)"
echo "   GPU Status: $(python -c 'import tensorflow as tf; print("Available" if len(tf.config.list_physical_devices("GPU")) > 0 else "Not Available")' 2>/dev/null)"

echo ""
echo "🚀 Ready for PhaseNet Indonesia training with GPU!"
echo "   Untuk training: python phasenet/train_indonesia.py [options]"
echo "   Untuk testing: python phasenet/test_indonesia.py [options]" 