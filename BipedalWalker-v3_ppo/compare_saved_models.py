"""
獨立的模型比較工具：直接載入已經存好的模型（.pkl 純 policy 或 .zip 完整 PPO），
在一組難度下各跑固定局數評估，比較平均報酬——不用重跑一次融合/訓練流程。

用法範例（單一難度）：
    python compare_saved_models.py --child ./models/ties_test_child.pkl \
        --dad ./best_model/p_models/model3.zip \
        --mom ./best_model/p_models/model2.zip \
        --difficulties 0.0 --n_eval 30

用法範例（難度掃描，逗號分隔）：
    python compare_saved_models.py --child ./models/ties_test_child.pkl \
        --dad ./best_model/p_models/model3.zip \
        --mom ./best_model/p_models/model2.zip \
        --difficulties 0.0,0.2,0.4,0.6,0.8,1.0 --n_eval 20
"""
import argparse
import random

import gym
import env.custom_env  # noqa: F401  # 確保 BipedalWalkerCustom-v0 的 register() 被執行
import cloudpickle
from stable_baselines3 import PPO


def load_any(path: str, env) -> PPO:
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            policy = cloudpickle.load(f)
        model = PPO("MlpPolicy", env, verbose=0)
        model.policy = policy.to(model.device)
    else:
        load_path = path[:-4] if path.endswith(".zip") else path
        model = PPO.load(load_path, env=env, device="cpu")
    return model


def evaluate(model: PPO, label: str, env, n_eval: int) -> tuple:
    rewards = []
    for ep in range(n_eval):
        seed = random.randint(0, 2**31 - 1)
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total += r
        rewards.append(total)
        print(f"  [{label}] ep{ep+1:02d}  seed={seed}  reward={total:.1f}")
    avg = sum(rewards) / len(rewards)
    std = (sum((r - avg) ** 2 for r in rewards) / len(rewards)) ** 0.5
    print(f"  [{label}] 平均 = {avg:.2f}  標準差 = {std:.2f}\n")
    return avg, std


def main():
    parser = argparse.ArgumentParser(description="在一組難度下比較已存檔的 child 模型跟父母模型的平均報酬")
    parser.add_argument("--child", type=str, required=True, help="child 模型路徑 (.pkl 或 .zip)")
    parser.add_argument("--dad", type=str, required=True, help="dad 模型路徑 (.pkl 或 .zip)")
    parser.add_argument("--mom", type=str, required=True, help="mom 模型路徑 (.pkl 或 .zip)")
    parser.add_argument("--env_id", type=str, default="BipedalWalkerCustom-v0")
    parser.add_argument("--difficulties", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                         help="逗號分隔的難度列表，預設掃描 0.0 到 1.0（可只給單一值，例如 0.0）")
    parser.add_argument("--n_eval", type=int, default=20, help="每個模型、每個難度各跑幾個 episode（預設 20）")
    args = parser.parse_args()

    difficulties = [float(d) for d in args.difficulties.split(",")]

    env = gym.make(args.env_id, difficulty=difficulties[0], render_mode=None)

    print(f"\n載入 child：{args.child}")
    child = load_any(args.child, env)
    print(f"載入 dad：{args.dad}")
    dad = load_any(args.dad, env)
    print(f"載入 mom：{args.mom}")
    mom = load_any(args.mom, env)

    results = {}  # {difficulty: {"dad": (avg, std), "mom": (...), "child": (...)}}

    for diff in difficulties:
        env.unwrapped.difficulty = diff
        print(f"\n===== 難度 {diff:.2f}，每個模型 {args.n_eval} 局 =====")
        print("===== Dad =====")
        dad_stats = evaluate(dad, "dad", env, args.n_eval)
        print("===== Mom =====")
        mom_stats = evaluate(mom, "mom", env, args.n_eval)
        print("===== Child =====")
        child_stats = evaluate(child, "child", env, args.n_eval)
        results[diff] = {"dad": dad_stats, "mom": mom_stats, "child": child_stats}

    env.close()

    # ── 彙整表 ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'難度':>6} | {'Dad':>14} | {'Mom':>14} | {'Child':>14} | 超越雙親？")
    print("-" * 72)
    win_count = 0
    for diff in difficulties:
        d = results[diff]
        dad_avg, dad_std = d["dad"]
        mom_avg, mom_std = d["mom"]
        child_avg, child_std = d["child"]
        better = child_avg > max(dad_avg, mom_avg)
        win_count += int(better)
        print(f"{diff:6.2f} | {dad_avg:6.2f}±{dad_std:5.2f} | {mom_avg:6.2f}±{mom_std:5.2f} | "
              f"{child_avg:6.2f}±{child_std:5.2f} | {'✅ Yes' if better else '❌ No'}")
    print("-" * 72)
    print(f"  Child 在 {len(difficulties)} 個難度中，贏過雙親 {win_count} 次")
    print("=" * 72)


if __name__ == "__main__":
    main()
