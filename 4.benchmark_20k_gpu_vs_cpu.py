import os
import time
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# 1. 20,160 专家架构单层定义（支持动态切换 Device 与 Host 模式）
# ----------------------------------------------------------------------
class MyriadBenchmarkLayer(nn.Module):
    def __init__(self, hidden_dim=1024, num_clusters=16, experts_per_cluster=45, rank=16, mode="host", device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.rank = rank

        # 文理双大核常驻显存
        self.big_arts = nn.Sequential(
            nn.Linear(hidden_dim, 3072, bias=False, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(3072, hidden_dim, bias=False, device=device, dtype=dtype)
        )
        self.big_sci = nn.Sequential(
            nn.Linear(hidden_dim, 3072, bias=False, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(3072, hidden_dim, bias=False, device=device, dtype=dtype)
        )

        # 调度器 (在 GPU)
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_cluster = nn.Linear(hidden_dim, num_clusters, bias=False, device=device, dtype=dtype)

        # 720 个微专家的权重张量 [16, 45, 16, 1024]
        raw_A = torch.randn(num_clusters, experts_per_cluster, rank, hidden_dim, dtype=dtype) * 0.02
        raw_B = torch.randn(num_clusters, experts_per_cluster, hidden_dim, rank, dtype=dtype) * 0.02

        if mode == "cuda":
            # 模式 A：全部直接塞在显存 (零 PCIe 传输)
            self.lora_A = raw_A.to(device)
            self.lora_B = raw_B.to(device)
        else:
            # 模式 B：全部放在主机 CPU 锁页内存 (按需流式搬运)
            self.lora_A_cpu = raw_A.pin_memory()
            self.lora_B_cpu = raw_B.pin_memory()
            self.transfer_stream = torch.cuda.Stream(device=device)

    def forward(self, x):
        # 1. 宏观双大核
        w_big = torch.softmax(self.router_big(x), dim=-1)
        big_out = (w_big[..., 0:1] * self.big_arts(x)) + (w_big[..., 1:2] * self.big_sci(x))

        # 2. 宗门路由
        w_cluster = torch.softmax(self.router_cluster(x), dim=-1)
        top2_scores, top2_idx = torch.topk(w_cluster, k=2, dim=-1)
        c1 = top2_idx[0, 0, 0].item()
        c2 = top2_idx[0, 0, 1].item()

        if self.mode == "cuda":
            # 显存直读
            A1, B1 = self.lora_A[c1], self.lora_B[c1]
            A2, B2 = self.lora_A[c2], self.lora_B[c2]
        else:
            # CPU 锁页内存 -> GPU PCIe 异步搬运
            with torch.cuda.stream(self.transfer_stream):
                A1 = self.lora_A_cpu[c1].to(self.device, non_blocking=True)
                B1 = self.lora_B_cpu[c1].to(self.device, non_blocking=True)
                A2 = self.lora_A_cpu[c2].to(self.device, non_blocking=True)
                B2 = self.lora_B_cpu[c2].to(self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # 3. 宗门微专家张量收缩
        h1 = torch.einsum('bsd,erd->bser', x, A1)
        out1 = torch.sum(torch.einsum('bser,edr->bsed', h1, B1), dim=2) / 45.0

        h2 = torch.einsum('bsd,erd->bser', x, A2)
        out2 = torch.sum(torch.einsum('bser,edr->bsed', h2, B2), dim=2) / 45.0

        micro_out = top2_scores[..., 0:1] * out1 + top2_scores[..., 1:2] * out2
        return big_out + 0.3 * micro_out


# ----------------------------------------------------------------------
# 2. 基准压测主流程 (28 层全模型等效仿真)
# ----------------------------------------------------------------------
def run_benchmark(mode, num_layers=28, num_tokens=100, warmup_tokens=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_vram = torch.cuda.memory_allocated() / (1024 ** 2)

    print(f"\n[*] 正在构建模式 [{mode.upper()}]: 28 层 x 720 专家 = 20,160 个微专家...")
    layers = nn.ModuleList([
        MyriadBenchmarkLayer(hidden_dim=1024, num_clusters=16, experts_per_cluster=45, rank=16, mode=mode, device="cuda:0")
        for _ in range(num_layers)
    ])

    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    model_vram_cost = allocated_vram - base_vram

    dummy_input = torch.randn(1, 1, 1024, device="cuda:0", dtype=torch.bfloat16)

    # 预热 (Warmup)
    print(f"[*] 正在进行 {warmup_tokens} 轮预热...")
    with torch.no_grad():
        for _ in range(warmup_tokens):
            h = dummy_input
            for layer in layers:
                h = layer(h)
    torch.cuda.synchronize()

    # 精确测时
    print(f"[*] 正在执行 {num_tokens} 次连续 Token 前向生成压测...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(num_tokens):
            h = dummy_input
            for layer in layers:
                h = layer(h)
        end_event.record()

    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    avg_latency_ms = total_time_ms / num_tokens
    throughput = (num_tokens / total_time_ms) * 1000.0

    return {
        "mode": mode,
        "vram_cost_mb": model_vram_cost,
        "peak_vram_mb": peak_vram,
        "avg_latency_ms": avg_latency_ms,
        "throughput": throughput
    }

def main():
    print("=" * 75)
    print("🏁【Myriad-MoE 20,160 微专家】CUDA 显存 vs CPU 内存 速度对决基准测试")
    print("=" * 75)

    # 1. 跑全显存模式 (CUDA)
    res_cuda = run_benchmark(mode="cuda", num_layers=28, num_tokens=100)
    
    # 2. 跑内存流式模式 (CPU Host DMA)
    res_cpu = run_benchmark(mode="host", num_layers=28, num_tokens=100)

    # 3. 打印对决终极战报
    print("\n" + "═" * 75)
    print("📊【20,160 专家：全显存 (CUDA) vs 内存流式 (CPU) 性能战报】")
    print("═" * 75)
    print(f"{'指标 / 维度':<25} | {'模式 A: 全放显存 (CUDA)':<20} | {'模式 B: 内存流式 (Host CPU)':<20}")
    print("─" * 75)
    print(f"{'静态占用显存 (VRAM)':<23} | {res_cuda['vram_cost_mb']:>14.2f} MB | {res_cpu['vram_cost_mb']:>16.2f} MB")
    print(f"{'生成峰值显存 (Peak)':<23} | {res_cuda['peak_vram_mb']:>14.2f} MB | {res_cpu['peak_vram_mb']:>16.2f} MB")
    print(f"{'单 Token 耗时 (Latency)':<20} | {res_cuda['avg_latency_ms']:>15.2f} ms | {res_cpu['avg_latency_ms']:>17.2f} ms")
    print(f"{'生成吞吐量 (Throughput)':<20} | {res_cuda['throughput']:>12.2f} tok/s | {res_cpu['throughput']:>14.2f} tok/s")
    print("═" * 75)

    vram_saved = res_cuda['vram_cost_mb'] - res_cpu['vram_cost_mb']
    speed_loss_pct = ((res_cuda['throughput'] - res_cpu['throughput']) / res_cuda['throughput']) * 100.0

    print("💡【架构权衡分析】:")
    print(f"   💾 显存节省量 : 净省 {vram_saved:.2f} MB ({vram_saved / 1024:.2f} GB)")
    print(f"   ⚡ 速度性能差 : 内存流式模式慢了 {speed_loss_pct:.1f}%")
    print(f"   🎯 战略结论   : 如果显卡是 24G/32G 选 [CUDA 模式] 享受极致极速；")
    print(f"                  如果显卡是 4G/8G 甚至手机芯片，选 [CPU 模式] 用极小速度代价突破显存墙！")
    print("═" * 75)

if __name__ == "__main__":
    main()
