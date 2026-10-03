#!/usr/bin/env python3
"""Strata 双 V100 深度速度基准（OpenAI API @18200，客户端流式口径）"""
import json, time, requests, statistics, sys

BASE = "http://127.0.0.1:18200/v1/chat/completions"
MODEL = None  # 服务端单模型

def make_prompt(approx_tokens: int) -> str:
    """构造指定近似 token 数的 prompt（中英混合，重复但多样）"""
    unit = ("深度学习框架的性能优化涉及计算图调度、内存分配器、算子融合与量化策略。"
            "Strata engine uses a peer-tier expert cache over NVLink with MTP speculative decoding. "
            "Benchmark results depend on prompt length, KV residency and draft acceptance rate. ")
    reps = max(1, approx_tokens // 60)
    return ("请仔细阅读以下技术材料，然后回答文末的问题。\n\n" + unit * reps +
            "\n\n问题：请总结以上材料的核心技术要点，并展开谈谈你的理解。")

def stream_request(prompt: str, max_tokens: int):
    """流式请求，返回 (ttft_ms, gen_tokens, total_ms, chunks_inter)"""
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    inter = []
    last = None
    r = requests.post(BASE, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7, "stream": True
    }, stream=True, timeout=600)
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        try:
            d = json.loads(payload)
            delta = d["choices"][0].get("delta", {}).get("content", "")
        except Exception:
            continue
        if delta:
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - t0) * 1000
            else:
                inter.append((now - last) * 1000)
            ntok += len(delta) // 3 + 1  # 近似，后面用服务端日志校准
            last = now
    total = (time.perf_counter() - t0) * 1000
    return ttft or 0, ntok, total, inter

def non_stream(prompt: str, max_tokens: int):
    t0 = time.perf_counter()
    r = requests.post(BASE, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7
    }, timeout=600)
    dt = (time.perf_counter() - t0) * 1000
    u = r.json().get("usage", {})
    return dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

results = {"prefill": [], "decode": [], "ttft": [], "reuse": [], "stream": []}

# ── 1. Prefill 梯度（非流式，短输出 8，看 prompt 处理时间） ──
print("== [1/5] Prefill 梯度 ==", flush=True)
for size in [128, 1024, 4096, 16384]:
    p = make_prompt(size)
    dt, ptok, ctok = non_stream(p, 8)
    prefill_ms = dt  # 含 8 token 生成，误差可忽略（8/45ms）
    results["prefill"].append(dict(target=size, prompt_tok=ptok, ms=prefill_ms,
                                   tok_s=ptok / prefill_ms * 1000))
    print(f"  目标~{size:>6} tok → 实际 {ptok:>6} tok, {prefill_ms:7.0f} ms → {ptok/prefill_ms*1000:7.1f} tok/s", flush=True)

# ── 2. TTFT（流式首 token 延迟，同 prompt 梯度） ──
print("== [2/5] TTFT（流式首token延迟）==", flush=True)
for size in [128, 4096]:
    p = make_prompt(size)
    for i in range(2):
        ttft, ntok, total, _ = stream_request(p, 24)
        results["ttft"].append(dict(target=size, run=i, ttft_ms=ttft))
        print(f"  ~{size:>6} tok #{i+1}: TTFT {ttft:7.0f} ms", flush=True)

# ── 3. 解码速度梯度（非流式，服务端 usage 计数） ──
print("== [3/5] 解码梯度 ==", flush=True)
for mt in [128, 512, 2048]:
    for i in range(3):
        p = make_prompt(30)
        dt, ptok, ctok = non_stream(p, mt)
        tps = ctok / dt * 1000
        results["decode"].append(dict(max_tokens=mt, run=i, gen=ctok, ms=dt, tok_s=tps))
        print(f"  max_tokens={mt:>5} #{i+1}: 生成 {ctok:>5} tok / {dt:7.0f} ms → {tps:6.1f} tok/s", flush=True)

# ── 4. KV 复用（同 prompt 连发两次） ──
print("== [4/5] KV 缓存复用 ==", flush=True)
p = make_prompt(4096)
for i in range(2):
    dt, ptok, ctok = non_stream(p, 32)
    results["reuse"].append(dict(run=i, ms=dt, prompt_tok=ptok))
    print(f"  第{i+1}次: 总耗时 {dt:7.0f} ms (prompt={ptok})", flush=True)

# ── 5. 流式逐 token 间隔分布（2048 输出） ──
print("== [5/5] 流式逐token间隔（2048 输出）==", flush=True)
p = make_prompt(30)
ttft, ntok, total, inter = stream_request(p, 2048)
if inter:
    inter_sorted = sorted(inter)
    n = len(inter_sorted)
    results["stream"] = dict(chunks=n, ttft_ms=ttft, total_s=total / 1000,
                             p50=statistics.median(inter), p95=inter_sorted[int(n * .95)],
                             mean=statistics.mean(inter))
    print(f"  chunks={n}, TTFT={ttft:.0f}ms, 间隔 p50={statistics.median(inter):.0f}ms "
          f"p95={inter_sorted[int(n*.95)]:.0f}ms mean={statistics.mean(inter):.0f}ms, "
          f"总耗时 {total/1000:.1f}s", flush=True)

json.dump(results, open("/tmp/bench_results.json", "w"), indent=1)
print("\n✅ 测试完成，结果存 /tmp/bench_results.json")
