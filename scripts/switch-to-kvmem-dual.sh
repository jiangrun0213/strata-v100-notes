#!/usr/bin/env bash
# 一键切换到 KVMem 双卡 TP2 模式（512K 上下文 · q8 KV · 张量并行测试前置）
# 用法: bash ~/switch-to-kvmem-dual.sh
set -x

# 1) 停掉用户级幽灵服务（它带 Restart 会无限复活 llama-server）
systemctl --user disable --now llama.service

# 2) 清掉所有残留 llama 进程
ps aux | grep '[l]lama-server' | awk '{print $2}' | xargs -r kill -9
sleep 4

# 3) 确认 GPU 清空（应显示 5 MiB 左右）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 4) 启动 KVMem 双卡 layer 模式（512K · q8 KV · CUDA0+CUDA1）
B=/home/jiangrun02123/kvmem-src/build/bin
M=/home/jiangrun02123/models
cd /home/jiangrun02123/kvmem-src
LD_LIBRARY_PATH=$B CUDA_VISIBLE_DEVICES=0,1 nohup $B/llama-kvmem-server \
    -m $M/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
    --mmproj $M/mmproj-Qwen3.8-27B-BF16.gguf \
    --no-mmproj-offload --image-max-tokens 512 \
    --gpu-layers all \
    --host 127.0.0.1 --port 18200 \
    -c 524288 -n 16384 \
    --kvmem --kvmem-budget 36864 --kvmem-gen-reserve 16384 \
    --kvmem-block-tokens 128 --kvmem-query-policy user \
    --kvmem-mtp-state snapshots \
    -ctk q8_0 -ctv q8_0 \
    -dev CUDA0,CUDA1 -sm layer -ts 1,1 \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --enable-thinking --reasoning-budget 4096 \
    > /tmp/kvmem_dual_final.log 2>&1 &
echo "KVMem 启动中（约 30-60 秒）..."

# 5) 等待并验证
sleep 60
tail -14 /tmp/kvmem_dual_final.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
curl -s --max-time 5 http://127.0.0.1:18200/health && echo " [KVMem 就绪]"
