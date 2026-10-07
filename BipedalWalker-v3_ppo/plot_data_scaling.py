# -*- coding: utf-8 -*-
"""
蒸餾資料量 vs 融合表現的趨勢圖（有 OT / 沒有 OT），資料來自 eval_per_difficulty.py --seed 2026
存下的每局 reward（同一批 1,100 張地形）。

左：11 難度平均 reward；右：OT 的效果（有 OT − 沒有 OT，逐局配對）與 95% 信賴區間。

⚠️ 5% 那一點的「沒有 OT」是 v2 #1（trial 35 參數、alpha=0），跟其他資料量的對照組
（inter_ot #18 參數、alpha=0）參數不完全相同，圖上用空心點標示。

用法：
  python plot_data_scaling.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPISODES = "./results/eval_per_difficulty_seed2026_episodes.json"
OUT_PNG = "./results/data_scaling_trend.png"

# (資料比例 %, 有 OT 的 key, 沒有 OT 的 key, 沒有 OT 是否為同參數對照)
POINTS = [
    (0.06, "scale_0p06pct_ot", "scale_0p06pct_no_ot", True),
    (0.2,  "scale_0p2pct_ot",  "scale_0p2pct_no_ot",  True),
    (1.0,  "scale_1pct_ot",    "scale_1pct_no_ot",    True),
    (5.0,  "v2_t18_preppo",    "v2_t1_no_ot",         False),
]
C_OT, C_NO, C_DAD, C_MOM = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

e = json.load(open(EPISODES, encoding="utf-8"))
diffs = sorted(e["dad"], key=float)
mid = [d for d in diffs if 0.6 <= float(d) <= 0.9]
avg = lambda k: float(np.mean([np.mean(e[k][d]) for d in diffs]))


def paired(a, b, ds):
    x = np.concatenate([np.array(e[a][d]) - np.array(e[b][d]) for d in ds])
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), 1.96 * se


xs = [p[0] for p in POINTS]
ot = [avg(p[1]) for p in POINTS]
no = [avg(p[2]) for p in POINTS]
eff_all = [paired(p[1], p[2], diffs) for p in POINTS]
eff_mid = [paired(p[1], p[2], mid) for p in POINTS]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans TC", "PingFang TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)

# ── 左：平均 reward ───────────────────────────────────────────────
a1.axhline(avg("dad"), color=C_DAD, lw=1.4, ls="--", zorder=1)
a1.axhline(avg("mom"), color=C_MOM, lw=1.4, ls="--", zorder=1)
a1.text(0.05, avg("dad") + 1.5, f"dad {avg('dad'):.1f}", color="#4b525b", fontsize=9)
a1.text(0.05, avg("mom") - 6, f"mom {avg('mom'):.1f}", color="#4b525b", fontsize=9)
a1.plot(xs, ot, color=C_OT, lw=2.6, marker="o", ms=7, mec="white", mew=1.4, label="有 OT", zorder=3)
a1.plot(xs[:3], no[:3], color=C_NO, lw=2.2, marker="o", ms=7, mec="white", mew=1.4, label="沒有 OT", zorder=2)
a1.plot(xs[2:], no[2:], color=C_NO, lw=2.2, ls=":", zorder=2)
a1.plot([xs[3]], [no[3]], marker="o", ms=7, mfc="white", mec=C_NO, mew=2, ls="none", zorder=3,
        label="沒有 OT（5%：參數略不同）")
for x, y in zip(xs, ot):
    a1.annotate(f"{y:.1f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=C_OT)
for x, y in zip(xs, no):
    a1.annotate(f"{y:.1f}", (x, y), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9, color=C_NO)
a1.set_xscale("log")
a1.set_xticks(xs)
a1.set_xticklabels(["0.06%\n4,346 筆", "0.2%\n14,487 筆", "1%\n72,438 筆", "5%\n362,193 筆"])
a1.set_ylim(205, 285)
a1.set_xlabel("蒸餾資料量（對數刻度）")
a1.set_ylabel("11 難度平均 reward")
a1.set_title("資料量 vs 融合表現（PPO 前）", loc="left", fontsize=12)
a1.legend(frameon=False, loc="lower right", fontsize=9)

# ── 右：OT 的效果與 95% CI ───────────────────────────────────────
off = 1.06
for ys, lab, col, shift in ((eff_all, "11 難度整體", C_OT, 1 / off), (eff_mid, "難度 0.6～0.9", "#4a3aa7", off)):
    xx = [x * shift for x in xs]
    m = [v[0] for v in ys]; ci = [v[1] for v in ys]
    a2.errorbar(xx[:3], m[:3], yerr=ci[:3], color=col, lw=2, marker="o", ms=7, mec="white", mew=1.4,
                capsize=4, label=lab, zorder=3)
    a2.errorbar([xx[3]], [m[3]], yerr=[ci[3]], color=col, marker="o", ms=7, mfc="white", mec=col, mew=2,
                capsize=4, ls="none", zorder=3)
    for x, y, c in zip(xx, m, ci):
        sig = (y - c) > 0 or (y + c) < 0
        a2.annotate(f"{y:+.1f}{' *' if sig else ''}", (x, y + c), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color=col)
a2.axhline(0, color="#7a818a", lw=1)
a2.set_xscale("log")
a2.set_xticks(xs)
a2.set_xticklabels(["0.06%", "0.2%", "1%", "5%"])
a2.set_xlabel("蒸餾資料量（對數刻度）")
a2.set_ylabel("有 OT - 沒有 OT（reward，逐局配對）")
a2.set_title("OT 的效果（誤差線 = 95% 信賴區間，* = 顯著）", loc="left", fontsize=12)
a2.legend(frameon=False, loc="upper right", fontsize=9)

for ax in (a1, a2):
    ax.grid(axis="y", color="#e6e9ec", lw=1)
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.text(0.01, 0.005, "seed 2026，11 難度 × 100 局，同一批地形；每種設定只有一個訓練出來的模型。空心點：5% 的「沒有 OT」為 v2 #1（trial 35 參數），與其他資料量的對照組參數不同。",
         fontsize=8.5, color="#7a818a")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT_PNG)
print(f"✅ {OUT_PNG}")
