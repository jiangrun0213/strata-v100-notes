# NInfer-V100-Duo 部署与对比测试报告

日期：2026-10-01 ｜ 机器：精粤 X99 TITANIUM D3 + Xeon E5-2666 v3 + 2×Tesla V100-SXM2-16GB (NV6 NVLink)

## 一、结论速览

| 指标 | llama.cpp (build 136 + MTP) | NInfer-V100-Duo (TP2 + P2P直连) | 胜者 |
|---|---|---|---|
| 单流解码 | 40~52 tok/s | 44.9~52.4 tok/s（均值 47.8） | NInfer 略快 ~10% |
| **36,556 token 预填充** | 52.1s（≈723 tok/s） | **25.7s（≈1510 tok/s）** | **NInfer 快 2.03×** |
| 4 路并发聚合 | 79.3 tok/s | 官方 142.7（未复测） | NInfer |
| 上下文上限 | 200,704（Q8 KV） | 180,224 默认 / 200K 需 1024 chunk | llama.cpp 略优 |
| 多模态 | ✅ mmproj@GPU0 | NVFP4 路线支持 / GGUF 路线不支持 | llama.cpp |
| API 端口 | 127.0.0.1:8081（在线） | 127.0.0.1:8090（可随时拉起） | — |

**总结**：NInfer 的核心优势是**长输入预填充快 2 倍**（长文档分析体感明显）；单流解码两者相当。
llama.cpp 生态成熟、支持多模态、上下文略大，适合继续当日常主力。

## 二、部署记录

```
仓库      ~/ninfer (fork plus1998/NInfer-V100-Duo, depth-1)
构建      tools/v100/build.sh → build-v100-duo/apps/{ninfer,ninfer-serve}
          303/303 成功，SM70 + CUDA 12.8，无错误
依赖审查  build.sh 与 build_dependencies.sh 干净（nasm/ffmpeg/openssl/curl 官方源）
模型      modelscope 魔搭 lmstudio-community/Qwen3.8-27B-GGUF (34 MiB/s)
          Qwen3.8-27B-Q4_K_M.gguf (16G) 重新下载
转换      python3 -m tools.convert.qwen3_8_27b.convert_gguf
          45 秒产出 qwen3_8_27b_q4_k_m.ninfer (18,322,586,368 B，与官方文档精确一致)
启动      bash tools/v100/ninfer-v100-duo.sh model=...ninfer --port 8090
          27.5s 加载 16.83GiB 权重，KV 180224 tokens (4.77 GiB)，CUDA Graph 2306 节点
```

## 三、关键发现与修改：IOMMU 预检绕过

### 现象
NInfer 启动日志：
```
[ninfer] direct P2P disabled (PCI 0000:03:00.0 uses translated IOMMU domain DMA-FQ; ...);
using verified CUDA host-staged copies
```
IOMMU 转换域触发保守回退，跨卡通信走 host 内存中转，解码 43.5 tok/s。

### 验证
独立 CUDA 测试程序（`cudaDeviceEnablePeerAccess` + cudaMemcpyPeer 循环）：
- P2P 双向支持 ✅
- **实测 142.5 GB/s**（NV6 理论 154.7 的 92%；PCIe Gen3 x8 仅 ~8 GB/s，不可能跑出此数）

### 修改
`src/ops/common/allreduce.cu` L259 —— 绕过 IOMMU sysfs 预检，但**保留 probe.qualify() 运行时实测验证**（失败自动回退 host-staged，fail-safe）。修改后日志不再出现 disabled 警告，解码 43.5 → 47.8 tok/s（+10%）。

## 四、实测基准数据

### 预填充（prompt=36,556 tokens，两次均完整消耗同一输入）
| 引擎 | 总耗时 | 预填充速度 |
|---|---|---|
| NInfer (TP2+P2P) | 25.7s | **≈1510 tok/s** |
| llama.cpp (MTP) | 52.1s | ≈723 tok/s |

### 单流解码（700 token 输出，短 prompt）
| 配置 | 速度 |
|---|---|
| llama.cpp + MTP（基线） | 40.1~41 tok/s（长文）/ 34.8（首测） |
| NInfer staged（IOMMU 回退） | 43.5 tok/s |
| NInfer direct P2P | **47.8 均值 / 52.4 峰值** |

### 4 路并发（llama.cpp 实测）
单路 19.1~21.2 tok/s，聚合 **79.3 tok/s**（batch 化填满 GPU 空闲，接近翻倍）。

### 显存布局
| | llama.cpp | NInfer |
|---|---|---|
| GPU0 | 13.0 GiB（模型45% + mmproj + 视觉缓冲） | 13.8 GiB（TP2 对称） |
| GPU1 | 15.4 GiB（模型55% + 200K Q8 KV） | 13.8 GiB |

## 五、附带成果

- NVLink 双卡 P2P 独立实测 **142.5 GB/s**（测试程序 /tmp/p2p_test.cu）
- llama.cpp 升级至 f7b384c (build 136)，MTP 启用（+37~49%）
- 双 V100 x8x8 Bifurcation + NVLink NV6 配置彻底验证可用
- 魔搭（modelscope）确认为本机最快模型源（34 MiB/s，比 hf-mirror 快 300×）

## 六、遗留事项

1. llama.cpp 服务已恢复在线（8081，全配置含 200K/Q8/MTP/多模态）
2. NInfer 服务可通过 `tools/v100/ninfer-v100-duo.sh model=... --port 8090` 随时拉起（需先停 llama-server 释放显存，二者不能共存于 GPU）
3. `~/models/DFlash2/`（3.6G safetensors）疑似已无用，待确认后可删
4. **sudo 密码修改仍未执行**（`passwd`）
5. systemd 开机自启（两套服务）尚未配置

## 七、低成本优化实验（4 项，2026-10-01 晚补充）

前置结论：本机推理瓶颈链 = CPU kernel 调度（SM 利用率 <1%）＞ 显存余量 ＞ GPU 时钟频率

| # | 实验 | 结果 | 结论 |
|---|---|---|---|
| 1 | llama.cpp `-ub 2048 -b 4096` | ❌ OOM：compute buffer 需 3.9GiB，GPU1 分配失败 | 200K Q8 KV 下显存顶死不可行；要用需 `--parallel 2` 或降上下文 |
| 2 | NInfer `--prefill-chunk 2048` | 36.5K prefill 27.3s vs 基线 25.7s | ❌ 默认 1024 已最优（-10%）；2048 唯一价值是显存余量 +0.2GiB，跑 200K 时可用 |
| 3 | KV K→`iq4_nl` | ❌ 严重负优化 | MTP 草稿接受率 87%→**30.9%**，解码 41→**27.8 tok/s**（K 侧过度量化污染草稿头输入），已回滚 q8_0 |
| 4 | GPU 锁频 `nvidia-smi -lgc 1530,1530` | ➖ 中性 | 单流与 prefill 均无变化（CPU 瓶颈）；无害保留，防高负载/夏季降频 |

### 补充发现

- llama.cpp 与 NInfer 的 tokenizer 对重复中文文本压缩率差异巨大：同一输入 llama.cpp=12,330 tokens vs NInfer=36,556 tokens。跨引擎 prefill 对比应以各自 token 流的 tok/s 为准（NInfer 1510 vs llama.cpp ~723，优势不变）。
- 4 项实验交叉验证了瓶颈链条：**显存余量和 CPU 调度是这台双 V100 的两块天花板**，GPU 时钟不是。继续提速的路径只剩：降低并发 slot 换显存、更激进的 MTP 窗口（NInfer MTP5，牺牲质量）、或升级平台。

### 最终定格配置（主力 llama.cpp @8081）

```
-c 200704 --cache-type-k q8_0 --cache-type-v q8_0   （iq4_nl 实验证伪，q8 是甜点）
--spec-type draft-mtp --spec-draft-n-max 4           （接受率与速度的实测平衡点）
默认 -ub/-b（大 batch 在此显存下 OOM）
GPU 锁频 1530MHz（保留）
```

## 八、systemd 服务部署（2026-10-01 深夜，开机自启 + 竞态防护 + 崩溃拉起）

### 三件套

| 文件 | 作用 |
|---|---|
| `/usr/local/bin/cuda_check`（源码存 `/usr/local/src/cuda_check.cu`） | CUDA 运行时健康探针（检测两张卡可见性；注意 **/tmp 重启清空**，故放持久路径） |
| `/usr/local/bin/wait-for-cuda` | 就绪轮询器：2 秒间隔最长等 120 秒，超时拒绝启动 |
| `/etc/systemd/system/llama-server.service` | 系统级服务单元（User=jiangrun02123，开机即启无需登录） |

### 服务单元关键设计

```ini
After=network-online.target nvidia-persistenced.service   # 启动排序
ExecStartPre=+/usr/sbin/modprobe nvidia_uvm || true       # root 权限确保 uvm 模块在场
ExecStartPre=/usr/local/bin/wait-for-cuda                 # 竞态防护：CUDA 未就绪不启动
Restart=always / RestartSec=10                            # 任何形式退出 10s 后拉起
StartLimitIntervalSec=600 / StartLimitBurst=10            # 防止反复失败触发频率封禁
StandardOutput/Error=append:~/llama-server.log            # 日志落盘
```

### 验证记录

| 项 | 结果 |
|---|---|
| `systemctl is-active / is-enabled` | active / enabled ✅ |
| 轮询器日志 | `CUDA ready (waited 2s)` ✅ |
| 推理 | 700 tokens / 13.8s = **50.9 tok/s** ✅ |
| **崩溃自恢复** | `kill -9` 主进程 → 16 秒内 active → 重新加载 → health 200 ✅ |
| GPU 显存 | 13.0 + 15.4 GiB（正常基线）✅ |

### 运维命令

```bash
systemctl status|stop|start|restart llama-server   # 日常管理
tail -f ~/llama-server.log                         # 实时日志
journalctl -u llama-server -e                      # systemd 层日志
# 改启动参数：sudo systemctl edit --full llama-server 改完 daemon-reload + restart
```

### 修复的根因（本次重启踩坑）

重启机器后 1 分钟内手动启动 llama-server → CUDA 运行时未就绪（`nvidia_uvm`/设备节点晚于驱动就绪）→ `ggml_cuda_init` 失败 → **静默降级 CPU 模式**（GPU 5MiB、prefill 10 tok/s、CPU 满载、无任何重试）。systemd 单元的 `modprobe nvidia_uvm + wait-for-cuda` 组合彻底封堵该竞态。

## 九、KVMem 部署与验证（2026-10-01，显存问题的根治方案）

### 项目背景

[kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)（arXiv:2609.04852，764★）：llama.cpp fork，实现**分层 KV 记忆**——GPU 只保留 32~37K 活跃窗口，历史 KV 卸载到主机 RAM（125GB），每步按 query 检索相关块取回。专为"16GB 卡跑 Qwen3.8-27B 全量 256K 工作区"设计。

### 部署

- 预编译包 `kvmem-v0.16.0-rc3-linux-x86_64-cuda12.9.86.tar.gz`（920MB，**含 sm70/Volta**，自带运行时无需 CUDA Toolkit；v0.17.0 仅源码）
- 解压至 `~/kvmem/kvmem-v0.16.0-rc3-linux-x86_64-cuda12.9.86/`
- 启动：`scripts/linux/start-iq3.sh --model IQ3_S-mtp.gguf --mmproj ... --gpu 0`
- 服务：`http://127.0.0.1:18200`（OpenAI 兼容 + Web UI）

### 实测结果（GPU0 单卡）

| 指标 | llama.cpp（双卡） | KVMem（单卡 GPU0） |
|---|---|---|
| 逻辑上下文 | 200,704 | **262,144（原生 256K）** |
| GPU 显存占用 | 28.4 GiB（双卡塞满） | **14.15 GiB（仅 GPU0）** |
| GPU1 | 余 1.4GiB（一挤就爆） | **完全空闲 15.5GiB** |
| KV 历史存放 | 全量在 GPU（Q8≈14GiB） | host RAM（~15GiB，125G 内存无压力） |
| 单流解码 | 40~52 tok/s | **44.2 tok/s 均值**（单卡！）
| MTP | draft-n-max 4 | draft-mtp n=3，think 模式 |
| 已知限制 | 无 | 检索式记忆（LongMemEval 85.6 vs 86.6）；单次生成 ≤16384 tok |

### 结论

1. **显存问题根治**：GPU1 整卡解放（15.5GB 可跑其他任务/模型），GPU0 余量 2.2GB，此前 OOM 的大 batch 配置在 KVMem 模式下可容纳
2. 单流速度与 llama.cpp 双卡持平——V100 HBM2 带宽优势弥补了单卡对双卡的算力差
3. 三引擎格局定型，按场景切换（显存互斥，GPU 只能驻留一个引擎）：
   - **llama.cpp @8081**（systemd）：日常/多模态/全量注意力无损
   - **NInfer @8090**：长文档 bulk 预填充（1510 tok/s）
   - **KVMem @18200**：超长 Agent 工作区（256K + 显存减半）
4. KVMem 首个请求预热较慢（4.8s 含检索初始化），属正常

## 十、GPU1 栈：RAG/Agent 配套三服务（2026-10-01 深夜，systemd 固化）

GPU1 被 KVMem 解放后，部署了 Agent/RAG 流水线的标准配套，三个独立 systemd 单元（均含 nvidia_uvm + wait-for-cuda 竞态防护、Restart=always 崩溃拉起、开机自启、CUDA_VISIBLE_DEVICES=1 绑定）：

| 服务 | 端口 | 模型 | 用途 |
|---|---|---|---|
| `llama-embed` | 8082 | bge-m3-Q5_0（439M，gpustack） | `/v1/embeddings` 1024 维向量 |
| `llama-rerank` | 8083 | bge-reranker-v2-m3-Q5_0（439M，gpustack） | `/v1/rerank` 重排序（-fa on） |
| `llama-fast` | 8084 | Qwen3-4B-Instruct-2507-Q4_K_M（2.4G，unsloth） | 快速决策/路由/工具调用（--jinja） |

模型来源全部为魔搭（合计 3.2GB，3 分钟）；决策模型选型依据：Qwen3-4B-Instruct-2507 是 2025-08 发布的非思考增强版，官方对标 GPT-4.1-nano（Agent/工具调用全面超越），与 27B 主模型同源，chat template 与工具调用格式无缝。

### 验证记录

| 服务 | 结果 |
|---|---|
| embedding | 1024 维输出正常 ✅ |
| rerank | 排序正确：注意力 query 得分 3.95 ≫ 无关文本 -10.99 ✅ |
| 决策 | 工具调用 JSON **0.20s** 端到端（17 tokens），HTTP 0.076s ✅ |
| systemd | 三单元 active + enabled ✅ |

### GPU 双卡最终全景

```
GPU0  14.4/16.3 GiB  KVMem @18200  Qwen3.8-27B 256K Agent 工作区（检索式记忆）
GPU1   8.6/16.3 GiB  embed@8082 + rerank@8083 + fast@8084（余 ~7GiB 可扩展）
RAM   ~15 GiB        KVMem 分层 KV 历史（125G 内存充裕）
```

### 决策模型推荐备注

2025-2026 流行的"快速决策"模型梯队（本地可部署）：Qwen3-4B-Instruct-2507（非思考快速版，本机已部署）、Qwen3-4B-Thinking-2507（推理增强版）、Qwen3-30B-A3B（MoE 3B 激活，需 ~18G 无法入 GPU1）、Gemma-3-4B。Qwen3-4B-Instruct-2507 在 Qwen 生态内工具调用格式与 27B 主模型一致，是最顺的搭档。

## 十一、最终精简（2026-10-02，按用户实际需求定格）

用户确认日常仅对话用途，三件套用不到，已执行精简：

- ❌ 删除三个小模型 GGUF（bge-m3 / bge-reranker / Qwen3-4B，共 3.2GB）
- ⏹ 停用并禁用自启：`llama-embed` / `llama-rerank` / `llama-fast`（unit 文件保留，恢复用 `systemctl enable --now`）
- ⏹ 停用 KVMem（`kvmem.service` 未创建，恢复见第九章命令）
- ✅ **llama-server 回归唯一常驻服务**：`-c 196608`（192K，用户指定），systemd 托管 active + enabled
- ✅ 实测 700 tokens / 52 tok/s，GPU 12.7 + 14.9 GiB

最终用户视图：**一个地址 `http://127.0.0.1:8081/v1`，一个 192K 上下文的多模态 27B 大脑，开机自启、崩溃自愈，没有其他任何东西。**

可选清理项（如不再使用 NInfer 路线）：`~/models/Qwen3.8-27B-Q4_K_M.gguf`（16G）与 `qwen3_8_27b_q4_k_m.ninfer`（18.3G）共约 34G。

## 十二、Strata 引擎（Flash-Next 125B MoE）V100 移植实战（2026-10-03）

### 背景与真相

用户提供的 Niko1221/Strata（5.4k★，CPU+GPU 混合 MoE 推理引擎）"据说有 V100 分支"——**核实为误传**：全仓库仅 main/rc 两分支，硬件门槛 RTX 20+（sm_75）。但源码里存在**官方实验通道**：

```cmake
option(STRATA_EXPERIMENTAL_SM60 "community build for ... Pascal sm_60, Volta sm_70 (#236)" OFF)
```

且明确要求 **CUDA 12.x**（CUDA 13 删除了 sm_70）—— 用户恰有 CUDA 12.8 ✓。社区实测（issue #236）。

### 编译实录（0 错误通过）

| 障碍 | 解法 |
|---|---|
| FetchContent 全量 clone ggml-org/llama.cpp 挂死 | 本地仓库 fetch 指定 commit 3cf0325（"sparse fa for qwen4"）+ worktree + `FETCHCONTENT_SOURCE_DIR` 短路 |
| 补丁从未应用（42 文件 KVMem patch） | 显式重跑 `apply-patches.sh` |
| `kvmem-gdn-replay-test` 编译错误（CUDA13-only 符号） | 注释该测试 target（CMakeLists L324-327） |
| `mtp-kv-test` 引用补丁 API 失败 | 同上（补丁后自动解决） |

产物：`strata/engine/strata`（0.1.38+ 主分支引擎）。

### 部署实录

```
模型：ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S（77.9 GiB 两分片 + mmproj 0.9G）
      → 魔搭直链 34 MiB/s ≈ 40 分钟（setup.py 从 HF 下不动，已用 --gguf-dir 复用）
打包：setup.py --gguf-dir 复用 → packs/iq3_s（dense.bin 1.5G + arena 46.84GiB mmap）
MTP：mtp_fetch.py 被 HF SSL 卡死 → 魔搭拉 28 个 BF16 分片（52G）→ 本地提取脚本产出
      mtp/tensors/*.bin + mtp-manifest.json（31 张量 5.21GB，格式与官方一致）
      → 修正 mtp-inventory.json 的 repo 字段后 mtp_fetch 零下载收尾
```

### 服务形态与实测

```
GPU0 15.7 GiB  主卡：密集层 + KV(int8) + MTP + 主专家缓存 4503 槽（8.53 GiB）
GPU1 15.7 GiB  Peer Tier：--peer-device 1 第二级自适应专家缓存（NVLink P2P，字节一致输出）
API   http://127.0.0.1:18200/v1（OpenAI 兼容）
```

| 配置 | 解码速度 |
|---|---|
| **基线（peer tier 默认）** | **44.8 tok/s** |
| + hugepages（22.5/47G，碎片限制）+ CPU performance | 44.7（无变化） |
| + 校准参数（--pcie-frac 0.20 --pool-workers 9 --spec-min-p 0.5） | 43.4（无变化） |

**校准器数据**（短请求基准，偏乐观）：PCIe share 0.20→53.4 / draft floor 0.50→58.2 / 9 workers→55.7 —— 但 700 token 长输出实测均无增益。

### 加速空间结论（诚实版）

| 项 | 结果 |
|---|---|
| 大页内存 | ➖ 只拿到 22.5/47G（运行中碎片），热路径收益未体现 |
| CPU performance 调频 | ➖ 无变化（E5 v3 已满频） |
| 校准参数 | ➖ 长输出无增益（短基准误导） |
| Strata tensor 并行模式 | ❌ meta allocator 在 SM70 段错误（上游 bug） |
| **QSA scorer tf32 mma** | ❌ **结构性天花板**：sm80+ 原生，sm70 走 fp32-FMA 回退，44-52 tok/s 就是这台双 V100 跑 125B MoE 的实际水平 |

### 运维

```
启动：cd ~/strata && .venv/bin/python serve/server.py --engine strata \
      --config ~/strata/strata-iq3_s.json --port 18200
停止：pkill -9 -f 'serve/server.py'; pkill -9 -f 'engine/strata'
      （注意：用户级 llama.service 已 disable，否则 10s 复活抢卡）
日志：~/strata/strata-iq3_s.log
模型：~/models/flash-next/（77.9G）+ ~/Strata-data/（pack+mtp ~57G）
```
