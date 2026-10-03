# Strata on 2× V100-SXM2 — Flash-Next 125B MoE 加载实录与深度基准

在一台 **双 Tesla V100-SXM2-16GB（NVLink NV6）** 的工作站上，成功编译并运行
[Niko1221/Strata](https://github.com/Niko1221/Strata)（CPU+GPU 混合 MoE 推理引擎），
加载 **Qwen3.8-Flash-Next-125B（IQ3_S，77.9 GiB）**，双卡各占 15.7 GiB，
解码平台 **59~63 tok/s**，Prefill 765 tok/s。

> Strata 官方硬件门槛是 RTX 20+（sm_75），并无 V100 支持。
> 本文档记录如何走通源码里的**实验通道**在 Volta (sm_70) 上跑起来，以及全套实测数据。

---

## 1. 背景：为什么 V100 能跑 Strata

Strata 源码中存在社区实验构建选项（对应 issue #236）：

```cmake
option(STRATA_EXPERIMENTAL_SM60 "community build for ... Pascal sm_60, Volta sm_70 (#236)" OFF)
```

两个硬性前提：

1. **必须 CUDA 12.x** —— CUDA 13 已删除 sm_70 架构支持（本机用 CUDA 12.8）
2. 编译时开启 `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`

硬件：精粤 X99 TITANIUM D3 + Xeon E5-2666 v3 + 2×V100-SXM2-16GB（BIOS **x8x8 Bifurcation**，NV6 NVLink）。

## 2. 移植过程实录

### 2.1 编译（0 错误通过，约 30-60 分钟）

产物：`strata/engine/strata`（v0.17.0 主分支引擎）。

| 障碍 | 解法 |
|---|---|
| FetchContent 全量 clone ggml-org/llama.cpp 挂死（网络+体积） | 本地仓库 fetch 指定 commit `3cf0325`（"CUDA: enable sparse fa for qwen4"）+ git worktree + `-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP` 短路，不再联网 |
| 42 文件 KVMem 补丁从未应用 | 显式重跑 `apply-patches.sh` |
| `kvmem-gdn-replay-test` 编译错误（CUDA13-only 符号） | 注释该测试 target |
| `mtp-kv-test` 引用补丁 API 失败 | 补丁应用后自动解决 |

核心编译命令（完整见 `scripts/build-strata-v100.sh`）：

```bash
# 前置：本地准备 llama.cpp worktree 短路 FetchContent
git clone https://github.com/ggml-org/llama.cpp strata-llamacpp-src
cd strata-llamacpp-src && git checkout 3cf0325

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<本地 llama.cpp worktree>
cmake --build build -j10
```

### 2.2 模型准备

**主模型（GGUF 77.9 GiB，2 分片 + mmproj 0.9G）**

- `setup.py` 自带下载走 HF，国内网络被 SSL 卡死 → 改从**魔搭 modelscope 直链**下载
  （实测 34 MiB/s，约 40 分钟），再用 `--gguf-dir` 复用本地文件打包：
  `python3 setup.py --gguf-dir <gguf目录>` → 产出 `packs/iq3_s`（dense.bin 1.5G + arena 46.84GiB mmap）

**MTP 草稿头（31 张量 / 5.21 GB）**

`mtp_fetch.py` 官方工具从 HF 下载被 SSL 卡死。绕行方案：

1. 从魔搭拉取 Qwen/Qwen3.8-Flash-Next 的 **28 个 BF16 safetensors 分片**（52G）
2. 运行 `scripts/mtp_extract_local.py`：读 `model.safetensors.index.json` 的 weight_map
   过滤 `mtp.*` 张量 → 从分片文件 seek/拷贝字节区间 → 产出 `mtp/tensors/*.bin` +
   `mtp-manifest.json`（31 张量带 sha256，格式与官方工具完全一致）
3. 修正 `mtp-inventory.json` 的 `repo` 字段后，`mtp_fetch` 校验通过零下载收尾

### 2.3 双卡启动（peer-tier 模式）

```bash
cd ~/strata && .venv/bin/python serve/server.py --engine strata \
    --config configs/strata-iq3_s.json --port 18200
```

引擎关键参数（完整见 `configs/strata-iq3_s.json`）：

```
--pack packs/iq3_s                    打包后的密集层+专家arena(mmap)
--native / --ple-gguf                 GGUF 分片
--expert-cache auto                   自适应专家缓存
--spec 4                              MTP 推测解码窗口 4
--mtp mtp/rt                          MTP 草稿头运行时
--max-context 262144                  256K 上下文
--kv int8 --kv-resident 32768         int8 KV，活跃驻留 32K
--peer-device 1                       GPU1 作为 peer tier（二级专家缓存，NVLink P2P）
```

加载完成后的显存形态：

```
GPU0  15.7 GiB  主卡：密集层 + KV(int8) + MTP + 主专家缓存 4503 槽（8.53 GiB）
GPU1  15.7 GiB  Peer Tier：二级自适应专家缓存（NVLink P2P 取回，字节一致输出）
API    http://127.0.0.1:18200/v1（OpenAI 兼容）
```

## 3. 深度速度基准

口径：客户端 OpenAI API 实测 + 服务端日志双口径交叉验证；MTP 接受率 70.5%（基准）↔ 71.5%（日常）互相印证。

### 3.1 结论速览

| 指标 | 结果 |
|---|---|
| **解码平台期（512/2048 tok 输出）** | **59~63 tok/s，波动 <5%** |
| 解码峰值（128 tok 短输出热身后） | 82 tok/s |
| **Prefill（14.5K tokens）** | **765 tok/s** |
| KV 复用（同 prompt 第二次） | **8017 ms → 58 ms（138×）**，99.9% 命中 |
| 流式 chunk 间隔 | p50 ≈ 0 ms / p95 = 65 ms |
| MTP 草稿接受率 | 70.5~71.5% |

### 3.2 Prefill 梯度（冷缓存）

| 目标长度 | 实际 prompt | 服务端耗时 | 吞吐 |
|---|---|---|---|
| ~128 | 186 tok | 4060 ms | 45.8 tok/s * |
| ~1K | 981 tok | 6342 ms | 154.7 tok/s |
| ~4K | 3684 tok | 8009 ms | **460 tok/s** |
| ~16K | 14549 tok | 19025 ms | **765 tok/s** |

\* 超短 prompt 的固定开销占主导，数字失真；真实含义是"小输入也要 ~3.5s 出首 token"。
**吞吐随 prompt 长度显著上升**（固定开销摊薄 + 批处理效率），长文档场景收益最大。

### 3.3 解码梯度（服务端口径，prompt=133 tok）

| max_tokens | 三次实测 (tok/s) | 均值 |
|---|---|---|
| 128 | 63.2 / 68.4 / **81.6** | 71.1（热身后走高） |
| 512 | 58.2 / 59.1 / 59.9 | 59.1 |
| 2048 | 62.6 / 62.1 / 60.5 | 61.7 |

**解码速度与输出长度无关** —— 512 与 2048 档完全一致，是稳定平台而非衰减曲线。

### 3.4 KV 缓存复用

| 请求 | 服务端 prefill 耗时 | 复用 |
|---|---|---|
| 同一 3684 tok prompt 第 1 次 | 8017 ms（全新读取） | 0/3684 |
| 第 2 次 | **58 ms** | **3679/3684 (99.9%)** |

多轮对话实测历史 KV 复用率 97.6% —— append-only 会话几乎零 prefill 成本。

### 3.5 流式体验（2048 tok 输出）

```
TTFT:            11.7 s（含 3.4s prefill + 首个大 batch）
chunk 间隔 p50:  0 ms（服务端每批推多个 token）｜ p95: 65 ms
整体:            1460 chunks / 32.5s ≈ 63 tok/s
```

### 3.6 三角互证

| 口径 | 速度 |
|---|---|
| 部署时固定 700 tok 基线 | 44.8 tok/s（含热身损失） |
| 本基准平台期 | **59~62 tok/s** |
| dsh-harness 日常使用加权（100 请求 / 40,038 tokens） | 59.2 tok/s |
| 短输出热身峰值 | 82 tok/s |

**59~63 tok/s 就是这台双 V100 跑 125B MoE 的真实平台。**

## 4. 加速空间结论（诚实版）

| 项 | 结果 |
|---|---|
| 大页内存 | ➖ 只拿到 22.5/47G（运行中碎片），热路径收益未体现 |
| CPU performance 调频 | ➖ 无变化（E5 v3 已满频） |
| 校准参数（--pcie-frac / --pool-workers / --spec-min-p） | ➖ 长输出实测无增益（短基准误导），保持默认 |
| Strata tensor 并行模式 | ❌ meta allocator 在 SM70 段错误（上游 bug）—— **只能用 peer-tier** |
| **QSA scorer tf32 mma** | ❌ **结构性天花板**：sm80+ 原生指令，sm70 走 fp32-FMA 回退，59~63 tok/s 已到头 |

## 5. 踩坑清单

| # | 坑 | 解法 |
|---|---|---|
| 1 | CUDA 13 无 sm_70 | 固定 CUDA 12.x（12.8 验证通过） |
| 2 | FetchContent 全量 clone llama.cpp 挂死 | 本地 worktree commit 3cf0325 + `FETCHCONTENT_SOURCE_DIR_*` 短路 |
| 3 | 42 文件 KVMem patch 未应用 | 显式重跑 `apply-patches.sh` |
| 4 | `kvmem-gdn-replay-test` 编译错误 | 注释该测试 target |
| 5 | `mtp_fetch.py` HF SSL 卡死 | 魔搭拉 BF16 分片 → 本地提取脚本 → 修正 inventory repo 字段 |
| 6 | tensor 并行 SM70 段错误 | 只用 `--peer-device` peer-tier 模式 |
| 7 | HF/魔搭下载渠道差异 | 魔搭直链 34 MiB/s，比 hf-mirror 快约 300× |
| 8 | `/tmp` 重启清空 | 编译产物/测试脚本/日志不放 /tmp |

## 6. 文件说明与复现

```
configs/strata-iq3_s.json     双卡启动配置（peer tier / int8 KV / 256K / MTP）
scripts/build-strata-v100.sh  SM70 实验构建编译脚本
scripts/mtp_extract_local.py  MTP 草稿头本地提取（从 BF16 safetensors 分片）
scripts/bench_strata.py       深度基准测试脚本（OpenAI 兼容 API）
```

```bash
# 基准测试复现（依赖: pip install requests）
python3 scripts/bench_strata.py

# 服务端口径原始记录
grep 'prompt [0-9]* tokens' ~/strata/strata-iq3_s.log

# 运维备忘
tail -f ~/strata/strata-iq3_s.log                        # 日志
pkill -9 -f 'serve/server.py'; pkill -9 -f 'engine/strata'  # 停止
# 磁盘: ~/models/flash-next/ 77.9G + ~/Strata-data/ ~57G
```

## 环境版本

- NVIDIA Driver 580.178.04 / CUDA 12.8 / Ubuntu 24.04
- Strata v0.17.0 源码 + llama.cpp 3cf0325（FetchContent 依赖）
- 模型：Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S（77.9 GiB）+ mmproj 0.9G

## License

MIT（本仓库笔记与脚本）。Strata 及其依赖遵循上游各自许可证。
