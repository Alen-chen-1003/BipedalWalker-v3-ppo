"""
每個難度各測 N 局，比較 dad / mom / distill_only / ot_progressive_iter。
用法：
  python eval_per_difficulty.py
"""
import os, sys, random, csv, argparse
import numpy as np
import gym
import cloudpickle

sys.path.insert(0, os.path.dirname(__file__))
import env.custom_env  # noqa: F401  — 註冊 BipedalWalkerCustom-v0

from stable_baselines3 import PPO

# ── 預設路徑 ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    "dad":          "./best_model/model2.zip",
    "mom":          "./best_model/model3.zip",
    "distill_only": "./models/ties_test_child_distill_only.pkl",
    "ot_prog_iter": "./models/ties_test_child_ot_progressive_iter_fixed.pkl",
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
            policy = cloudpickle.load(f)
        model = PPO("MlpPolicy", tmp_env, verbose=0)
        model.policy = policy.to(model.device)
    else:
        load_path = path[:-4] if path.endswith(".zip") else path
        model = PPO.load(load_path, env=tmp_env, device="cpu")
    tmp_env.close()
    return model

# ── 評估一個模型在單一難度 ────────────────────────────────────────────────────
def eval_at_difficulty(model, difficulty: float, n: int = N_PER_DIFF):
    rewards = []
    for _ in range(n):
        env = gym.make(ENV_ID, difficulty=difficulty)
        seed = random.randint(0, 2**31 - 1)
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
    args = parser.parse_args()

    print("載入模型中...")
    models = {}
    for name, path in DEFAULTS.items():
        if not os.path.exists(path):
            print(f"  ⚠️  找不到 {name}：{path}，跳過")
            continue
        models[name] = load_model(path)
        print(f"  ✅ {name} 載入完成")

    os.makedirs("./results", exist_ok=True)

    # CSV header
    fieldnames = ["difficulty"] + [f"{m}_mean" for m in models] + \
                 [f"{m}_std"  for m in models]
    rows = []

    for diff in sorted(args.diff):
        row = {"difficulty": diff}
        print(f"\n── 難度 {diff:.1f} ──")
        for name, model in models.items():
            mean, std, _ = eval_at_difficulty(model, diff, n=args.n)
            row[f"{name}_mean"] = round(mean, 2)
            row[f"{name}_std"]  = round(std,  2)
            print(f"  {name:20s}  {mean:7.2f} ± {std:.2f}")
        rows.append(row)

    # 寫 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ 結果已存至 {OUT_CSV}")

    # 總覽
    print("\n" + "="*60)
    print(f"{'難度':>6}", end="")
    for m in models:
        print(f"  {m:>20}", end="")
    print()
    for row in rows:
        print(f"{row['difficulty']:>6.1f}", end="")
        for m in models:
            print(f"  {row[f'{m}_mean']:>18.2f}", end="")
        print()

if __name__ == "__main__":
    main()
