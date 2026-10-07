# -*- coding: utf-8 -*-
"""
把 dad / mom / distill_only / inter_ot 在各難度的分數輸出成表格（CSV）和趨勢圖（PNG）。
資料來自 eval_per_difficulty.py --seed 2026 存下的每局 reward（同一批地形，每難度 100 局）。

inter_ot = v2 調參 trial #18（logs/optuna_ot_prog_iter_v2/trial_18_preppo.pkl，ot_frac 0.612）
inter_ot (ot_frac=1) = ot_frac 固定 1 的 v2 調參 trial #3（logs/optuna_ot_prog_iter_v2_otfrac1/trial_3_preppo.pkl）

用法：
  python plot_inter_ot_difficulty.py
"""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPISODES = "./results/eval_per_difficulty_seed2026_episodes.json"
OUT_CSV = "./results/inter_ot_per_difficulty.csv"
OUT_PNG = "./results/inter_ot_per_difficulty.png"

# 顯示名稱 → eval 裡的 key；顏色沿用 dataviz 參考調色盤，依固定順序取前五格
MODELS = [
    ("inter_ot",     "v2_t18_preppo", "#2a78d6"),
    ("inter_ot (ot_frac=1)", "otfrac1_t3_preppo", "#e87ba4"),
    ("distill_only", "distill_only",  "#eb6834"),
    ("dad",          "dad",           "#1baf7a"),
    ("mom",          "mom",           "#eda100"),
]

e = json.load(open(EPISODES, encoding="utf-8"))
diffs = sorted(e["dad"], key=float)
x = [float(d) for d in diffs]
mean = {n: [float(np.mean(e[k][d])) for d in diffs] for n, k, _ in MODELS}
std = {n: [float(np.std(e[k][d])) for d in diffs] for n, k, _ in MODELS}

# ── 表格 ──────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 直接開不會亂碼
    w = csv.writer(f)
    w.writerow(["difficulty"] + [f"{n}_mean" for n, _, _ in MODELS] + [f"{n}_std" for n, _, _ in MODELS])
    for i, d in enumerate(diffs):
        w.writerow([d] + [round(mean[n][i], 1) for n, _, _ in MODELS] + [round(std[n][i], 1) for n, _, _ in MODELS])
    w.writerow(["avg"] + [round(float(np.mean(mean[n])), 1) for n, _, _ in MODELS] + [""] * len(MODELS))
print(f"✅ 表格：{OUT_CSV}")

# ── 趨勢圖 ────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans TC", "PingFang TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
main = ("inter_ot", "inter_ot (ot_frac=1)")
for n, _, c in reversed(MODELS):  # 兩個 inter_ot 最後畫，壓在最上面
    ax.plot(x, mean[n], color=c, lw=2.6 if n in main else 2, marker="o", ms=5,
            markeredgecolor="white", markeredgewidth=1.2, label=n, zorder=3 if n in main else 2)
# 右端直接標名稱；終點太近的標籤往下推開，避免疊在一起
ends = sorted(((mean[n][-1], n) for n, _, _ in MODELS), reverse=True)
ys, gap = [], 7.5
for y, n in ends:
    ys.append(min(y, ys[-1] - gap) if ys else y)
for (y, n), yl in zip(ends, ys):
    ax.annotate(n, (x[-1], y), xytext=(x[-1] + 0.02, yl), textcoords="data",
                va="center", fontsize=9.5, color="#4b525b")
ax.set_xticks(x)
ax.set_xticklabels(diffs)
ax.set_xlim(-0.03, 1.2)
ax.set_ylim(100, 320)
ax.set_xlabel("地形難度")
ax.set_ylabel("平均 reward（每難度 100 局）")
ax.set_title("各難度平均 reward（seed 2026，五個模型同一批地形）", fontsize=12, loc="left")
ax.grid(axis="y", color="#e6e9ec", lw=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"✅ 趨勢圖：{OUT_PNG}")
