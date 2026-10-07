"""
每個難度各測 N 局，比較 dad / mom / distill_only / ot_progressive_iter。
用法：
  python eval_per_difficulty.py
"""
import os, sys, random, csv, argparse, json
import numpy as np
import gym
import cloudpickle

sys.path.insert(0, os.path.dirname(__file__))
import env.custom_env  # noqa: F401  — 註冊 BipedalWalkerCustom-v0

from stable_baselines3 import PPO

# ── 預設路徑 ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    "dad":           "./best_model/model2.zip",
    "mom":           "./best_model/model3.zip",
    "distill_only":  "./models/ties_test_child_distill_only.pkl",
    "ot_prog_iter":  "./models/ties_test_child_ot_progressive_iter_fixed.pkl",
    "ties_prog_iter": "./models/ties_test_child_ties_progressive_iter.pkl",
    "ot_paper_pure":       "./models/ties_test_child_ot_paper_pure.pkl",
    "ot_paper_cross":      "./models/ties_test_child_ot_paper_cross.pkl",
    "ot_paper_cross_pure": "./models/ties_test_child_ot_paper_cross_pure.pkl",
    # 論文式單一網路融合（完全照論文含欄位方向 bug）+ 全網蒸餾，兩種 PPO 步數
    "ot_paper_single_distill_100k": "./models/ties_test_child_ot_paper_single_paper_nobias_distill.pkl",
    "ot_paper_single_distill_1M":   "./models/ot_paper_single_faithful_distill_1M.pkl",
    # 修正版融合 + 不蒸餾 + hard-seed 複習 + 250 萬步 PPO
    "ot_paper_single_curric_2.5M": "./models/ot_paper_single_fixed_nodistill_curric_2.5M.pkl",
    # 差異化學習率（純通道梯度 ×0.2625）對照組
    "distill_only_difflr_1M":  "./models/distill_only_difflr_pure_1M.pkl",
    "ot_prog_iter_difflr":     "./models/ot_prog_iter_difflr_pure.pkl",
    # --evolved_all --crossover zero（難度 0.5 起跳、無複習、10 萬步、差異化 lr 壓純通道）
    "evolved_all_zero": "./models/child_model3_x_model2_fused_full.pkl",
    # ot_progressive_iter 套用 evolved_all 的訓練設定（0.5 起跳、10 萬步、差異化 lr）
    "ot_prog_iter_diff05": "./models/ot_prog_iter_diff05_100k.pkl",
    # tune_ot_progressive_iter.py 階段 1 存下的 PPO 前模型（蒸餾只用 5% 資料）：
    # trial 35 = Optuna 最高分，trial 0 = 原本預設參數（基準線）
    "optuna_t35_preppo": "./logs/optuna_ot_prog_iter/trial_35_preppo.pkl",
    "optuna_t0_preppo":  "./logs/optuna_ot_prog_iter/trial_0_preppo.pkl",
    # ablation_ot_vs_distill.py：跟 trial 35 同一批 5% 蒸餾資料的兩個對照組
    "ablation_t35_no_ot":        "./logs/optuna_ot_prog_iter/ablation_t35_no_ot.pkl",
    "ablation_distill_only_5pct": "./logs/optuna_ot_prog_iter/ablation_distill_only_5pct.pkl",
    # 新流程：只蒸餾這一層 + 鏈式 T（上一層的 T 影響下一層），及其無 OT 對照
    "ablation_chain_layer":       "./logs/optuna_ot_prog_iter/ablation_chain_layer.pkl",
    "ablation_chain_layer_no_ot": "./logs/optuna_ot_prog_iter/ablation_chain_layer_no_ot.pkl",
    # v2 調參（舊流程、縮小範圍、25 局評估、保留地形驗證）：#18/#26 驗證前 2 名，#1 = trial 35 參數關掉 OT
    "v2_t18_preppo": "./logs/optuna_ot_prog_iter_v2/trial_18_preppo.pkl",
    "v2_t26_preppo": "./logs/optuna_ot_prog_iter_v2/trial_26_preppo.pkl",
    "v2_t1_no_ot":   "./logs/optuna_ot_prog_iter_v2/trial_1_preppo.pkl",
    # ablation_data_scaling.py：inter_ot 參數，蒸餾資料 1% / 0.2% / 0.06%，有 OT / 沒有 OT；raw_concat＝直接拼接不訓練
    "raw_concat":           "./logs/data_scaling/raw_concat.pkl",
    "scale_1pct_ot":        "./logs/data_scaling/scale_1pct_ot.pkl",
    "scale_1pct_no_ot":     "./logs/data_scaling/scale_1pct_no_ot.pkl",
    "scale_0p2pct_ot":      "./logs/data_scaling/scale_0p2pct_ot.pkl",
    "scale_0p2pct_no_ot":   "./logs/data_scaling/scale_0p2pct_no_ot.pkl",
    "scale_0p06pct_ot":     "./logs/data_scaling/scale_0p06pct_ot.pkl",
    "scale_0p06pct_no_ot":  "./logs/data_scaling/scale_0p06pct_no_ot.pkl",
    "scale_1pct_distill_only":    "./logs/data_scaling/scale_1pct_distill_only.pkl",
    "scale_0p2pct_distill_only":  "./logs/data_scaling/scale_0p2pct_distill_only.pkl",
    "scale_0p06pct_distill_only": "./logs/data_scaling/scale_0p06pct_distill_only.pkl",
    # ot_frac 固定 1.0 的 v2 調參（ot_prog_iter_v2_otfrac1）：#3 = 保留地形驗證第 1 名
    "otfrac1_t3_preppo": "./logs/optuna_ot_prog_iter_v2_otfrac1/trial_3_preppo.pkl",
    # run_ot_env_rounds.py：每輪 OT 後改用環境 PPO 10 萬步（alpha 0.9→0.035、ot_frac 1）；
    # preppo = 16 輪跑完、最終微調前；final = 再加 100 萬步（難度從 0.5 起跳）
    "ot_env_rounds_preppo": "./logs/ot_env_rounds/preppo.pkl",
    "ot_env_rounds_final":  "./logs/ot_env_rounds/final.pkl",
}
ENV_ID       = "BipedalWalkerCustom-v0"
DIFFICULTIES = [round(x * 0.1, 1) for x in range(11)]   # 0.0, 0.1, … 1.0
N_PER_DIFF   = 100
OUT_CSV      = "./results/eval_per_difficulty.csv"

# ── 載入模型 ──────────────────────────────────────────────────────────────────
def load_model(path: str):
    tmp_env = gym.make(ENV_ID, difficulty=0.0)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = cloudpickle.load(f)
        # --test_ties 存的是 policy；--evolved_all 存的是整個 PPO agent，兩種都接
        policy = obj.policy if isinstance(obj, PPO) else obj
        model = PPO("MlpPolicy", tmp_env, verbose=0)
        model.policy = policy.to(model.device)
    else:
        load_path = path[:-4] if path.endswith(".zip") else path
        model = PPO.load(load_path, env=tmp_env, device="cpu")
    tmp_env.close()
    return model

# ── 評估一個模型在單一難度 ────────────────────────────────────────────────────
def fixed_seeds(base_seed: int, difficulty: float, n: int):
    """
    同一個 base_seed + 難度 → 永遠同一組 seed。每個模型都跑這組 seed，
    地形（reset(seed) 會重設 np_random）完全一樣，才能逐局配對比較。
    """
    rng = random.Random(base_seed * 1000 + int(round(difficulty * 10)))
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def eval_at_difficulty(model, difficulty: float, n: int = N_PER_DIFF, seeds=None):
    rewards = []
    for i in range(n):
        env = gym.make(ENV_ID, difficulty=difficulty)
        seed = random.randint(0, 2**31 - 1) if seeds is None else seeds[i]
        obs  = env.reset(seed=seed)
        if isinstance(obs, tuple):          # gym ≥0.26
            obs = obs[0]
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:          # gym ≥0.26
                obs, r, term, trunc, _ = step_out
                done = term or trunc
            else:
                obs, r, done, _ = step_out
            total += r
        rewards.append(total)
        env.close()
    return np.mean(rewards), np.std(rewards), rewards

# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,   default=N_PER_DIFF)
    parser.add_argument("--diff", type=float, nargs="*", default=DIFFICULTIES)
    parser.add_argument("--only", type=str, default=None,
                         help="逗號分隔的 DEFAULTS key，只評估這幾個模型（預設全部）；"
                              "用來在舊結果已經有其他模型時，只補跑新加的模型，其餘欄位維持舊值")
    parser.add_argument("--seed", type=int, default=None,
                         help="給了就改用固定 seed：每個模型在每個難度都跑同一組地形，才能公平比較。"
                              "結果另存 ./results/eval_per_difficulty_seed<seed>.csv（不跟舊的隨機 seed 結果混在一起），"
                              "每局 reward 另存同名 _episodes.json 供配對檢定。預設不給＝原本的隨機 seed 行為")
    args = parser.parse_args()

    global OUT_CSV
    episodes_path = None
    if args.seed is not None:
        OUT_CSV = f"./results/eval_per_difficulty_seed{args.seed}.csv"
        episodes_path = OUT_CSV.replace(".csv", "_episodes.json")

    only_keys = None if args.only is None else {k.strip() for k in args.only.split(",")}

    print("載入模型中...")
    models = {}
    for name, path in DEFAULTS.items():
        if only_keys is not None and name not in only_keys:
            continue
        if not os.path.exists(path):
            print(f"  ⚠️  找不到 {name}：{path}，跳過")
            continue
        models[name] = load_model(path)
        print(f"  ✅ {name} 載入完成")

    os.makedirs("./results", exist_ok=True)

    # 讀舊結果（如果有），用來跟這次新跑的模型合併，而不是覆蓋掉舊模型的欄位
    old_rows_by_diff = {}
    old_model_names = []
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                old_rows_by_diff[round(float(r["difficulty"]), 1)] = r
            if reader.fieldnames:
                old_model_names = sorted({
                    fn[:-5] for fn in reader.fieldnames if fn.endswith("_mean")
                })

    all_model_names = list(dict.fromkeys(old_model_names + list(models.keys())))
    fieldnames = ["difficulty"] + [f"{m}_mean" for m in all_model_names] + \
                 [f"{m}_std"  for m in all_model_names]
    rows = []
    episodes = {}
    if episodes_path is not None and os.path.exists(episodes_path):
        with open(episodes_path, encoding="utf-8") as f:
            episodes = json.load(f)

    for diff in sorted(args.diff):
        diff = round(diff, 1)
        row = dict(old_rows_by_diff.get(diff, {}))
        for key in list(row.keys()):
            if key.endswith("_mean") or key.endswith("_std"):
                # csv.DictReader 讀回來的是字串；空字串代表該模型還沒評估過，保持原樣
                row[key] = float(row[key]) if row[key] not in ("", None) else ""
        row["difficulty"] = diff
        print(f"\n── 難度 {diff:.1f} ──")
        for name, model in models.items():
            seeds = None if args.seed is None else fixed_seeds(args.seed, diff, args.n)
            mean, std, eps = eval_at_difficulty(model, diff, n=args.n, seeds=seeds)
            if episodes_path is not None:
                episodes.setdefault(name, {})[str(diff)] = [float(r) for r in eps]
            row[f"{name}_mean"] = round(mean, 2)
            row[f"{name}_std"]  = round(std,  2)
            print(f"  {name:20s}  {mean:7.2f} ± {std:.2f}")
        rows.append(row)

    # ⚠️ 只評估部分難度時（--diff），沒跑到的難度也必須把舊資料原樣寫回去，
    # 否則整份結果會被截斷成只剩這次跑的那幾行。
    done_diffs = {r["difficulty"] for r in rows}
    for d, old in old_rows_by_diff.items():
        if d in done_diffs:
            continue
        keep = dict(old)
        for key in list(keep.keys()):
            if key.endswith("_mean") or key.endswith("_std"):
                keep[key] = float(keep[key]) if keep[key] not in ("", None) else ""
        keep["difficulty"] = d
        rows.append(keep)
    rows.sort(key=lambda r: r["difficulty"])

    # 寫 CSV（合併後的完整結果，舊模型的欄位沒被選中評估的話原樣保留）
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ 結果已存至 {OUT_CSV}")
    if episodes_path is not None:
        with open(episodes_path, "w", encoding="utf-8") as f:
            json.dump(episodes, f)
        print(f"✅ 每局 reward 已存至 {episodes_path}")
    models = {name: None for name in all_model_names}  # 讓下面的總覽印出全部模型欄位，不只這次新跑的

    # 總覽
    print("\n" + "="*60)
    print(f"{'難度':>6}", end="")
    for m in models:
        print(f"  {m:>20}", end="")
    print()
    for row in rows:
        print(f"{row['difficulty']:>6.1f}", end="")
        for m in models:
            v = row.get(f'{m}_mean', "")
            print(f"  {v:>18.2f}" if v not in ("", None) else f"  {'-':>18}", end="")
        print()

if __name__ == "__main__":
    main()
