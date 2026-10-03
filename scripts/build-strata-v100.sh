#!/usr/bin/env bash
# Strata V100 (sm_70 实验构建) 编译脚本 v2
# 已完成的准备工作：
#   ✅ 源码就位 /home/jiangrun02123/strata (v0.17.0)
#   ✅ llama.cpp 子模块绕过：worktree /home/jiangrun02123/strata-llamacpp-src
#      (commit 3cf0325 "CUDA: enable sparse fa for qwen4"，FetchContent 不再需要联网)
# 用法: bash ~/build-strata-v100.sh
set -x
cd /home/jiangrun02123/strata

export PATH=/usr/local/cuda/bin:$PATH
unset https_proxy http_proxy   # 依赖已在本地，不需要代理

# 配置（FETCHCONTENT_SOURCE_DIR 直接指向本地 worktree，秒过）
rm -rf build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=/home/jiangrun02123/strata-llamacpp-src

# 编译（53 个 .cu + ggml，预计 30-60 分钟）
cmake --build build -j10
BUILD_EXIT=$?

echo "================ 产物 ================"
ls build/bin 2>/dev/null | grep -viE 'test' | head -20
echo "BUILD_EXIT=$BUILD_EXIT"
