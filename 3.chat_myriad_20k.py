from collections import Counter
import copy
import os
from threading import Thread
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

CLUSTER_NAMES = [
    "Code_Algo", "Code_DS", "Code_Debug", "Code_Arch", "Math_Algebra",
    "Math_Geo", "Math_Prob", "Math_Arith", "Sci_Physics", "Sci_Chem",
    "Sci_Biology", "Sci_Astronomy", "Arts_Rhetoric", "Arts_Philosophy",
    "Arts_Summary", "Arts_Chat"
]


class MyriadInferenceWrapper(nn.Module):

    def __init__(self,
                 original_mlp,
                 hidden_dim=1024,
                 num_clusters=16,
                 experts_per_cluster=45,
                 rank=16,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.rank = rank

        self.big_arts = original_mlp.to(device)
        self.big_sci = copy.deepcopy(original_mlp).to(device)

        self.router_big = nn.Linear(hidden_dim,
                                    2,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        self.router_cluster = nn.Linear(hidden_dim,
                                        num_clusters,
                                        bias=False,
                                        device=device,
                                        dtype=dtype)

        # 20,160 专家常驻 CPU 内存 (仅 1.23 GB)
        self.lora_A_cpu = None
        self.lora_B_cpu = None
        self.intra_router = None

        self.transfer_stream = torch.cuda.Stream(device=device)

        # 统计计数
        self.total_arts_weight = 0.0
        self.total_sci_weight = 0.0
        self.cluster_counter = Counter()

    def reset_stats(self):
        self.total_arts_weight = 0.0
        self.total_sci_weight = 0.0
        self.cluster_counter.clear()

    def forward(self, x):
        current_token = x[:, -1:, :]

        # 1. 双大核文理调度
        logits_big = self.router_big(current_token)
        w_big = torch.softmax(logits_big, dim=-1)
        self.total_arts_weight += w_big[0, 0, 0].item()
        self.total_sci_weight += w_big[0, 0, 1].item()

        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (w_big[..., 0:1] * arts_out) + (w_big[..., 1:2] * sci_out)

        # 2. 16 宗门路由
        logits_cluster = self.router_cluster(current_token)
        w_cluster = torch.softmax(logits_cluster, dim=-1)

        # 选出得分最高的 Top-2 宗门
        top2_scores, top2_clusters = torch.topk(w_cluster, k=2, dim=-1)
        c1 = top2_clusters[0, 0, 0].item()
        c2 = top2_clusters[0, 0, 1].item()
        self.cluster_counter[c1] += 1
        self.cluster_counter[c2] += 1

        # 3. 异步流式拉取对口宗门的专家切片 (仅拉取需要的微块，极速零拷贝)
        with torch.cuda.stream(self.transfer_stream):
            A1 = self.lora_A_cpu[c1].to(self.device, non_blocking=True)
            B1 = self.lora_B_cpu[c1].to(self.device, non_blocking=True)
            A2 = self.lora_A_cpu[c2].to(self.device, non_blocking=True)
            B2 = self.lora_B_cpu[c2].to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # 组内精算并融合
        # 宗门 1
        h1 = torch.einsum('bsd,erd->bser', x, A1)
        out1 = torch.sum(torch.einsum('bser,edr->bsed', h1, B1), dim=2) / 45.0

        # 宗门 2
        h2 = torch.einsum('bsd,erd->bser', x, A2)
        out2 = torch.sum(torch.einsum('bser,edr->bsed', h2, B2), dim=2) / 45.0

        micro_out = top2_scores[..., 0:1] * out1 + top2_scores[..., 1:2] * out2

        return big_out + 0.3 * micro_out


def show_myriad_dashboard(model):
    total_arts = sum(layer.mlp.total_arts_weight
                     for layer in model.model.layers)
    total_sci = sum(layer.mlp.total_sci_weight for layer in model.model.layers)
    all_big = total_arts + total_sci
    arts_pct = (total_arts / all_big * 100) if all_big > 0 else 50
    sci_pct = (total_sci / all_big * 100) if all_big > 0 else 50

    total_clusters = Counter()
    for layer in model.model.layers:
        total_clusters.update(layer.mlp.cluster_counter)

    print("\n" + "═" * 70)
    print("🌌【Myriad-MoE: 20,160 微专家宇宙全息透视】:")
    print(f"   🏛️  文科原版大核: {arts_pct:5.1f}% [{'█' * int(arts_pct // 5):<20}]")
    print(f"   🔬 理科特训大核: {sci_pct:5.1f}% [{'█' * int(sci_pct // 5):<20}]")
    print("─" * 70)
    print("🪐【16 宗门活跃热力图 (Top Active Clusters)】:")
    for cid, cnt in total_clusters.most_common(5):
        c_name = CLUSTER_NAMES[cid]
        print(f"   ✨ #{cid:02d} [{c_name:<16}]: 激活 {cnt:,} 次")
    print("═" * 70)


def main():
    model_id = "Qwen/Qwen3-0.6B"
    weights_path = "myriad_moe_20k_weights.pt"
    assert os.path.exists(
        weights_path), f"找不到权重文件 {weights_path}，请先执行训练！"

    print("=" * 70)
    print("🚀 正在唤醒【Myriad-MoE: 20,160 专家万象终端】...")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    for layer in model.model.layers:
        layer.mlp = MyriadInferenceWrapper(layer.mlp,
                                           hidden_dim=hidden_dim,
                                           num_clusters=16,
                                           experts_per_cluster=45,
                                           rank=16,
                                           device="cuda:0",
                                           dtype=dtype)

    print(f"[*] 正在挂载 20,160 个微专家到 CPU 锁页内存 (仅吃 1.23 GB RAM)...")
    saved = torch.load(weights_path, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.big_sci.load_state_dict(saved[f"layer_{i}_big_sci"])
        layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
        layer.mlp.router_cluster.load_state_dict(
            saved[f"layer_{i}_router_cluster"])
        layer.mlp.lora_A_cpu = saved[f"layer_{i}_lora_A"].pin_memory()
        layer.mlp.lora_B_cpu = saved[f"layer_{i}_lora_B"].pin_memory()

    print("\n✅ 两万微专家帝国已全员就位！")
    print("👉 提示：输入 'clear' 重置记忆，输入 'exit' 退出\n")

    messages = [{
        "role": "system",
        "content": "You are a master of all domains with 20,000 micro-experts."
    }]

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "clear":
            messages = [{
                "role":
                "system",
                "content":
                "You are a master of all domains with 20,000 micro-experts."
            }]
            print("🧹 记忆已重置。")
            continue

        for layer in model.model.layers:
            layer.mlp.reset_stats()

        messages.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=[tokenizer.eos_token_id, 151645])

        print("\n🤖 Assistant: ", end="", flush=True)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        t0 = time.perf_counter()
        thread.start()

        accumulated_text = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            accumulated_text += chunk
        thread.join()

        elapsed_sec = time.perf_counter() - t0
        gen_tokens = len(
            tokenizer.encode(accumulated_text, add_special_tokens=False))
        speed = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0
        print(
            f"\n\n⚡ 速度: {speed:.1f} tokens/s (共 {gen_tokens} 字, 耗时 {elapsed_sec*1000:.0f} ms)"
        )

        show_myriad_dashboard(model)
        messages.append({"role": "assistant", "content": accumulated_text})


if __name__ == "__main__":
    main()
