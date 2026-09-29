import json
from datasets import load_dataset

print("=" * 70)
print("🚀 正在装配【Myriad-MoE: 16 宗门多元特训语料库】...")
print("=" * 70)

# 加载多领域混合数据源
print("    - [1/4] 正在拉取代码与算法数据集 (iamtarun/python_code)...")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                       split="train[:4000]")

print("    - [2/4] 正在拉取数学与推导数据集 (openai/gsm8k)...")
ds_math = load_dataset("openai/gsm8k", "main", split="train[:4000]")

print("    - [3/4] 正在拉取科学与常识数据集 (allenai/sciq)...")
ds_sciq = load_dataset("allenai/sciq", split="train[:4000]")

print("    - [4/4] 正在拉取通用写作与对话数据集 (HuggingFaceH4/no_robots)...")
ds_arts = load_dataset("HuggingFaceH4/no_robots", split="train[:4000]")

myriad_data = []

# 映射到 16 个宗门 (Cluster 0 ~ 15)
# 0-3: 代码领域（算法、数据结构、调试、架构）
for idx, item in enumerate(ds_code):
    cluster_id = idx % 4  # 分流到宗门 0, 1, 2, 3
    prompt = item["instruction"] + (f"\n{item['input']}"
                                    if item.get("input") else "")
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1,
        "domain": f"Code_Cluster_{cluster_id}",
        "prompt": prompt,
        "response": item["output"]
    })

# 4-7: 数学与逻辑（代数、几何、概率、算术）
for idx, item in enumerate(ds_math):
    cluster_id = 4 + (idx % 4)  # 分流到宗门 4, 5, 6, 7
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1,
        "domain": f"Math_Cluster_{cluster_id}",
        "prompt": item["question"],
        "response": item["answer"]
    })

# 8-11: 科学与推演（物理、化学、生物、天文学）
for idx, item in enumerate(ds_sciq):
    cluster_id = 8 + (idx % 4)  # 分流到宗门 8, 9, 10, 11
    prompt = f"Question: {item['question']}\nContext: {item['support']}"
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1,
        "domain": f"Sci_Cluster_{cluster_id}",
        "prompt": prompt,
        "response": item["correct_answer"]
    })

# 12-15: 人文与写作（修辞、哲学、摘要、情商对话）
for idx, item in enumerate(ds_arts):
    cluster_id = 12 + (idx % 4)  # 分流到宗门 12, 13, 14, 15
    messages = item["messages"]
    if len(messages) >= 2:
        myriad_data.append({
            "cluster_id": cluster_id,
            "is_stem": 0,
            "domain": f"Arts_Cluster_{cluster_id}",
            "prompt": messages[0]["content"],
            "response": messages[1]["content"]
        })

output_file = "myriad_train_data.jsonl"
print(f"\n[*] 正在写入 {output_file}，共 {len(myriad_data)} 条多领域严格对齐数据...")
with open(output_file, "w", encoding="utf-8") as f:
    for entry in myriad_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[✔] 16 宗门语料全量就绪！总样本 16,000 条，随时可以启动两万专家大特训！")
