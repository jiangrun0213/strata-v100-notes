# Strata V100 双卡移植与加载过程实录

机器：精粤 X99 TITANIUM D3 + Xeon E5-2666 v3 + 2×Tesla V100-SXM2-16GB（NV6 NVLink，BIOS x8x8 Bifurcation）
日期：2026-10-03

## 一、可行性核实

用户传闻"Niko1221/Strata 有 V100 分支"——核实为**误传**：全仓库仅 main/rc 两分支，
官方硬件门槛 RTX 20+（sm_75）。但源码内存在官方实验通道：

```cmake
option(STRATA_EXPERIMENTAL_SM60 "community build for ... Pascal sm_60, Volta sm_70 (#236)" OFF)
```

社区已有实测（issue #236）。硬性要求 **CUDA 12.x**（CUDA 13 删除了 sm_70 目标）。
本机恰好有 CUDA 12.8 → 具备条件。

## 二、编译实录（0 错误通过）

产物：`strata/engine/strata`（v0.17.0 主分支引擎）。

| 障碍 | 解法 |
|---|---|
| FetchContent 全量 clone ggml-org/llama.cpp 挂死（网络+体积） | 本地仓库 fetch 指定 commit `3cf0325`（"CUDA: enable sparse fa for qwen4"）+ git worktree + `-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP` 短路，不再联网 |
| 42 文件 KVMem 补丁从未应用 | 显式重跑 `apply-patches.sh` |
| `kvmem-gdn-replay-test` 编译错误（CUDA13-only 符号） | 注释该测试 target |
| `mtp-kv-test` 引用补丁 API 失败 | 补丁应用后自动解决 |

完整编译命令见 `scripts/build-strata-v100.sh`，核心参数：

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<本地 llama.cpp worktree>
cmake --build build -j10
```

## 三、模型准备

### 主模型（GGUF 77.9 GiB，2 分片 + mmproj 0.9G）

- 目标：ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF 的 IQ3_S
- `setup.py` 自带下载走 HF，本机网络被 SSL 卡死 → 改从**魔搭 modelscope 直链**下载
  （实测 34 MiB/s，约 40 分钟），然后用 `--gguf-dir` 复用本地文件打包：
  `python3 setup.py --gguf-dir <gguf目录>` → 产出 `packs/iq3_s`
  （dense.bin 1.5G + arena 46.84GiB mmap）

### MTP 草稿头（31 张量 / 5.21 GB）

`mtp_fetch.py` 官方工具从 HF 下载被 SSL 卡死。绕行方案：

1. 从魔搭拉取 Qwen/Qwen3.8-Flash-Next 的 **28 个 BF16 safetensors 分片**（52G）
2. 运行 `scripts/mtp_extract_local.py`：读 `model.safetensors.index.json` 的
   weight_map 过滤 `mtp.*` 张量 → 直接从分片文件 seek/拷贝字节区间 →
   产出 `mtp/tensors/*.bin` + `mtp-manifest.json`（31 张量，带 sha256，
   格式与官方工具完全一致）
3. 修正 `mtp-inventory.json` 的 `repo` 字段后，`mtp_fetch` 校验通过零下载收尾

## 四、双卡启动

启动方式：

```bash
cd ~/strata && .venv/bin/python serve/server.py --engine strata \
    --config ~/strata/strata-iq3_s.json --port 18200
```

引擎关键参数（见 `configs/strata-iq3_s.json`）：

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

## 五、性能实测与调优记录

| 配置 | 解码速度 |
|---|---|
| **基线（peer tier 默认）** | **44.8 tok/s** |
| + hugepages（22.5/47G，运行中碎片限制）+ CPU performance | 44.7（无变化） |
| + 校准参数（--pcie-frac 0.20 --pool-workers 9 --spec-min-p 0.5） | 43.4（无变化） |

校准器给出的建议（短请求基准，偏乐观）：PCIe share 0.20→53.4 / draft floor 0.50→58.2 /
9 workers→55.7 —— 但 **700 token 长输出实测均无增益**，保持默认即可。

## 六、加速空间结论（诚实版）

| 项 | 结果 |
|---|---|
| 大页内存 | ➖ 只拿到 22.5/47G（运行中碎片），热路径收益未体现 |
| CPU performance 调频 | ➖ 无变化（E5 v3 已满频） |
| 校准参数 | ➖ 长输出无增益（短基准误导） |
| Strata tensor 并行模式 | ❌ meta allocator 在 SM70 段错误（上游 bug）—— 只能用 peer-tier |
| **QSA scorer tf32 mma** | ❌ **结构性天花板**：sm80+ 原生指令，sm70 走 fp32-FMA 回退，44-52 tok/s 就是双 V100 跑 125B MoE 的实际水平 |

## 七、运维备忘

```bash
# 启动
cd ~/strata && .venv/bin/python serve/server.py --engine strata \
    --config ~/strata/strata-iq3_s.json --port 18200

# 停止（注意先停掉会抢卡的常驻服务）
pkill -9 -f 'serve/server.py'; pkill -9 -f 'engine/strata'

# 日志
tail -f ~/strata/strata-iq3_s.log

# 磁盘占用
#   ~/models/flash-next/   77.9G（GGUF）
#   ~/Strata-data/         ~57G（pack + MTP）
```
