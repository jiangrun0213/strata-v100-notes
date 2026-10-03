# Strata on 2× V100-SXM2 — Flash-Next 125B MoE 加载实录

在一台 **双 Tesla V100-SXM2-16GB（NVLink NV6）** 的工作站上，
成功编译并运行 [Niko1221/Strata](https://github.com/Niko1221/Strata)（CPU+GPU 混合 MoE 推理引擎），
加载 **Qwen3.8-Flash-Next-125B（IQ3_S，77.9 GiB）**，双卡各占 15.7 GiB，解码 **44.8 tok/s**。

> Strata 官方硬件门槛是 RTX 20+（sm_75），并无 V100 支持。
> 本仓库记录如何走通源码里的**实验通道**在 Volta (sm_70) 上跑起来。

## 背景：为什么 V100 能跑 Strata

Strata 源码中存在社区实验构建选项（对应 issue #236）：

```cmake
option(STRATA_EXPERIMENTAL_SM60 "community build for ... Pascal sm_60, Volta sm_70 (#236)" OFF)
```

两个硬性前提：

1. **必须 CUDA 12.x** —— CUDA 13 已删除 sm_70 架构支持（本机用 CUDA 12.8）
2. 编译时开启 `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`

## 双卡运行形态（peer-tier 模式）

```
GPU0  15.7 GiB  主卡：密集层 + KV(int8, 32K resident) + MTP 草稿头 + 主专家缓存(4503 槽, 8.53 GiB)
GPU1  15.7 GiB  Peer Tier：--peer-device 1 二级自适应专家缓存（NVLink P2P 取回）
API   http://127.0.0.1:18200/v1（OpenAI 兼容）
```

| 配置 | 解码速度 |
|---|---|
| 基线（peer tier 默认） | **44.8 tok/s** |
| + hugepages / CPU performance / 校准参数 | 均无增益（44~45 持平） |

**性能天花板说明**：Strata 的 QSA scorer 依赖 sm_80+ 的 tf32 mma 指令，
sm_70 走 fp32-FMA 回退路径 —— 44~52 tok/s 就是双 V100 跑这个 125B MoE 的实际上限。
另：Strata 的 tensor 并行模式在 SM70 上有 meta allocator 段错误（上游 bug），
**只能用 peer-tier 模式**，不要开 TP。

## 文件说明

```
configs/strata-iq3_s.json     双卡启动配置（peer tier / int8 KV / 256K / MTP）
scripts/build-strata-v100.sh  SM70 实验构建编译脚本
scripts/mtp_extract_local.py  MTP 草稿头本地提取（从 BF16 safetensors 分片）
docs/porting-notes.md         完整过程实录：编译→模型→MTP→双卡启动→调优→踩坑
docs/benchmarks.md            深度速度基准（prefill/解码/KV复用/流式/MTP）
```

## 快速复现

### 1. 编译（约 30-60 分钟）

```bash
# 前置：本地准备 llama.cpp worktree 以短路 FetchContent（避免全量 clone 挂死）
git clone https://github.com/ggml-org/llama.cpp strata-llamacpp-src
cd strata-llamacpp-src && git checkout 3cf0325   # "CUDA: enable sparse fa for qwen4"

bash scripts/build-strata-v100.sh
# 产物: build/bin/strata
```

关键 cmake 参数见脚本：`STRATA_EXPERIMENTAL_SM60=ON` +
`FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<本地 worktree>`。

### 2. 模型打包 + MTP 草稿头

```bash
# 模型下载：国内环境强烈建议用魔搭 modelscope 直链（实测 34 MiB/s）
# GGUF (77.9 GiB, 2 分片) → setup.py 打包（复用已下载文件）
python3 setup.py --gguf-dir <gguf目录>          # → packs/iq3_s

# MTP 草稿头：mtp_fetch.py 从 HF 下载会被 SSL 卡死，
# 改为手动拉 BF16 safetensors 分片后本地提取：
python3 scripts/mtp_extract_local.py            # → 31 张量 / 5.21 GB
```

### 3. 双卡启动

```bash
cd ~/strata && .venv/bin/python serve/server.py --engine strata \
    --config configs/strata-iq3_s.json --port 18200
```

配置要点（完整见 `configs/strata-iq3_s.json`）：
`--peer-device 1`（GPU1 做 peer tier）、`--kv int8 --kv-resident 32768`、
`--spec 4`（MTP 窗口）、`--max-context 262144`。

## 踩坑清单

| # | 坑 | 解法 |
|---|---|---|
| 1 | CUDA 13 无 sm_70 | 固定 CUDA 12.x（12.8 验证通过） |
| 2 | FetchContent 全量 clone llama.cpp 挂死 | 本地 worktree 指定 commit 3cf0325 + `FETCHCONTENT_SOURCE_DIR_*` 短路 |
| 3 | 42 文件 KVMem patch 未应用 | 显式重跑 `apply-patches.sh` |
| 4 | `kvmem-gdn-replay-test` 编译错误（CUDA13-only 符号） | 注释该测试 target |
| 5 | `mtp_fetch.py` HF SSL 卡死 | 魔搭拉 28 个 BF16 分片（52G）→ 本地提取脚本 → 修正 `mtp-inventory.json` 的 repo 字段后零下载收尾 |
| 6 | tensor 并行模式 SM70 段错误（meta allocator 上游 bug） | 只用 `--peer-device` peer-tier 模式 |
| 7 | 大页内存只拿到 22.5/47G（运行中碎片） | 启动前预留，或接受默认；实测对热路径无收益 |
| 8 | 校准器建议参数（短基准偏乐观） | 长输出实测无增益，保持默认即可 |

## 环境版本

- NVIDIA Driver 580.178.04 / CUDA 12.8 / Ubuntu 24.04
- Strata v0.17.0 源码 + llama.cpp 3cf0325（FetchContent 依赖）
- 模型：Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S（77.9 GiB）+ mmproj 0.9G

## License

MIT（本仓库笔记与脚本）。Strata 及其依赖遵循上游各自许可证。
