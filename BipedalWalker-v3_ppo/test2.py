import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# === 檔案路徑 ===
json_path = r"C:\BipedalWalker-v3_ppo\BipedalWalker-v3_ppo\logs\ga_eval1\fitness_stage1.json"

# === 讀取 GA 結果 ===
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 初始父母 (model2~model13) ===
BASE_PARENTS = [f"model{i}" for i in range(2, 14)]

# === 計算初代父母平均 Reward ===
base_rewards = []
for p in BASE_PARENTS:
    for ext in [".zip", ".pkl"]:
        key = p + ext
        if key in data:
            base_rewards.append(float(data[key]["avg_reward"]))

if len(base_rewards) == 0:
    raise ValueError("未找到初始父母模型，請確認檔案命名是否包含 model2~model13")
base_mean = np.mean(base_rewards)
print(f"初始父母平均分數: {base_mean:.2f}")

# === 計算子代資料 ===
pairs = []
lineages = []
for name, vals in data.items():
    if re.match(r"model\d+", name):  # 跳過初始父母
        continue
    try:
        gen_match = re.search(r"child_gen(\d+)", name)
        gen = int(gen_match.group(1)) if gen_match else 0
        child_r = float(vals["avg_reward"])
        delta = child_r - base_mean
        pairs.append((gen, base_mean, child_r, delta))
        lineages.extend(re.findall(r"model\d+", name))
    except Exception:
        continue

pairs = np.array(pairs, dtype=float)
gene_count = Counter(lineages)
genes, counts = zip(*gene_count.most_common())

# === 繪圖 ===
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (6a) 初代 vs 子代表現
sns.scatterplot(
    x=pairs[:, 1], y=pairs[:, 2],
    hue=pairs[:, 0], palette="Spectral",
    s=80, ax=axes[0], edgecolor="black", linewidth=0.5
)
axes[0].plot([min(pairs[:,2]), max(pairs[:,2])],
             [min(pairs[:,2]), max(pairs[:,2])],
             "k--", lw=1.2, label="y=x baseline")
axes[0].axvline(base_mean, color="gray", linestyle="--", label="Base Parent Mean")
axes[0].set_xlabel("Base Parents Mean Reward (model2–13)")
axes[0].set_ylabel("Child Reward")
axes[0].set_title("(6a) Base Parent vs Child Performance")
axes[0].legend(title="Generation")

# (6b) ΔReward 分佈
sns.histplot(pairs[:,3], bins=20, kde=True, color="orange", ax=axes[1])
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("ΔReward (Child - Base Parent Mean)")
axes[1].set_ylabel("Count")
axes[1].set_title("(6b) ΔReward Distribution")

# (6c) 基因貢獻熱力圖
sns.heatmap(
    np.array(counts).reshape(1, -1),
    cmap="YlGnBu", annot=True, fmt="d",
    cbar=False, ax=axes[2],
    xticklabels=genes, yticklabels=["Frequency"]
)
axes[2].set_title("(6c) Gene Contribution Heatmap")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()