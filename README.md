# strata-v100-notes — 双 V100 推理平台实战笔记

在一台 **精粤 X99 TITANIUM D3 + Xeon E5-2666 v3 + 2×Tesla V100-SXM2-16GB (NVLink NV6)** 的工作站上，
部署、对比、调优四个推理引擎（llama.cpp / NInfer-V100-Duo / KVMem / Strata）的完整记录。

含：实测基准数据、编译移植踩坑、systemd 生产化部署、NVLink 带宽实测程序。

## 硬件要点

| 项 | 配置 |
|---|---|
| 平台 | X99 寨板 + E5-2666 v3（无核显） |
| GPU | 2× V100-SXM2-16GB，SXM2→PCIe 转接卡 |
| 关键配置 | BIOS **x8x8 Bifurcation** —— 单 x16 端口只能接一个设备，拆分后 CPU 端口裂变出第二个根端口，第二张卡才有归属 |
| NVLink | NV6 全链路 Active（6×25.78 GB/s），**实测 P2P 142.5 GB/s**（理论 154.7 的 92%） |

> 踩坑：SXM2 转接卡在 x8 电气下完全正常工作（最初误判为"只认 x16"，
> 真相是未拆分的 x16 端口物理上只暴露一个设备）。

## 四引擎实测对比（Qwen3.8-27B / Flash-Next-125B）

| 指标 | llama.cpp (MTP) | NInfer-V100-Duo (TP2) | KVMem (分层KV) | Strata (MoE) |
|---|---|---|---|---|
| 单流解码 | 40~52 tok/s | 47.8 均值 / 52.4 峰值 | 44.2 tok/s（**单卡**） | 44.8 tok/s（125B MoE） |
| 长输入预填充 | ≈723 tok/s | **≈1510 tok/s (2×)** | — | — |
| 上下文 | 192K~200K（Q8 KV） | 180K 默认 | **256K**（KV 卸载到 RAM） | 128K |
| 显存 | 双卡 28.4 GiB | 双卡 13.8 GiB 对称 | **仅 GPU0 14.2 GiB** | 双卡 15.7 GiB 对称 |
| 多模态 | ✅ mmproj | GGUF 路线不支持 | ✅ | ✅ |
| 适用场景 | 日常主力/生态最全 | 长文档 bulk 预填充 | 超长 Agent 工作区/解放第二张卡 | 125B MoE 推理 |

**核心结论**：
- 显存余量和 CPU 调度是这台双 V100 的两块天花板（GPU 时钟不是 —— 锁频实验证实中性）
- V100 (sm_70) 上的结构性限制：Strata 的 QSA tf32-mma 需要 sm_80+，sm_70 走 fp32-FMA 回退
- KV 量化甜点是 **q8_0**；iq4_nl 是严重负优化（MTP 草稿接受率 87%→31%）

## 目录

```
docs/
  NInfer-V100-Duo对比测试报告.md   ← 全量记录：部署/修改/基准/踩坑（12 章）
scripts/
  build-strata-v100.sh             Strata 编译脚本（SM70 + CUDA 12.8 实验通道）
  mtp_extract_local.py             MTP 草稿头本地提取（绕开 HF 下载卡死）
  switch-to-kvmem-dual.sh          KVMem 双卡 TP2 512K 启动脚本
  p2p_test.cu                      NVLink P2P 带宽实测程序
  systemd/
    llama-server.service           生产级服务单元（含竞态防护设计）
    wait-for-cuda                  CUDA 就绪轮询器
    cuda_check.cu                  CUDA 运行时健康探针
```

## 脚本用法

```bash
# NVLink 带宽实测（期望 ~142 GB/s；若只有 ~8 GB/s 说明 P2P 被禁用）
nvcc -o p2p_test scripts/p2p_test.cu && ./p2p_test

# Strata 编译（需 CUDA 12.x，CUDA 13 已删除 sm_70）
bash scripts/build-strata-v100.sh

# systemd 部署（开机自启 + 崩溃自愈 + CUDA 竞态防护）
sudo cp scripts/systemd/llama-server.service /etc/systemd/system/
sudo cp scripts/systemd/wait-for-cuda /usr/local/bin/
nvcc -o /usr/local/bin/cuda_check scripts/systemd/cuda_check.cu
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
```

## 关键踩坑清单（详见 docs 报告）

1. **PCIe Bifurcation**：双卡识别的前提。BIOS 拆 x16→x8x8 后拓扑多出独立根端口
2. **IOMMU 转换域**：NInfer 会因 sysfs 预检误判而禁用 P2P（回退 host 中转，-10%）。
   修改 `allreduce.cu` 保留运行时实测验证、只绕过静态预检，fail-safe
3. **CUDA 竞态**：开机 1 分钟内启动 llama-server → `nvidia_uvm` 未就绪 →
   **静默降级 CPU 模式**（无任何报错！）。systemd 单元用
   `ExecStartPre=+modprobe nvidia_uvm || true` + 就绪轮询器封堵
4. **MTP on sm_70**：NVIDIA 开源内核模块（`-open`）不支持 Volta，必须用专有驱动分支
5. **跨引擎 tokenizer 差异**：同一文本 llama.cpp=12.3K tokens vs NInfer=36.6K tokens，
   跨引擎 prefill 对比必须按各自 token 流计算
6. **`/tmp` 重启清空**：编译产物、测试程序、服务日志都不要放 /tmp
7. **模型下载**：魔搭 modelscope 直连 34 MiB/s，比 hf-mirror 快约 300×（国内环境）
8. **Strata 上游 bug**：tensor 并行模式的 meta allocator 在 SM70 段错误，只能用
   peer-tier 模式（NVLink P2P 专家缓存分级）

## 环境版本

- NVIDIA Driver 580.178.04 / CUDA 12.8 / Ubuntu 24.04
- llama.cpp f7b384c (build 136, GGML_CUDA=ON SM70)
- NInfer-V100-Duo fork plus1998/NInfer-V100-Duo (SM70, 303/303 构建通过)

## License

MIT（笔记与脚本）。引用的各上游项目遵循其各自许可证。
