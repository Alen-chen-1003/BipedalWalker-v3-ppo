# -*- coding: utf-8 -*-
"""
把 ot_paper_fuse.py 融合出來的模型丟回環境訓練，對應論文 cheetah.ipynb 的
`## With retraining`（481-487 行）。

論文那邊是 5e5 步，這裡預設 1e6 步，跟本專案其他實驗（ot_progressive_iter /
ties_progressive_iter / distill_only）的 --progressive_steps_per_round 對齊，
這樣結果才放得進同一張比較表。

用法：
  python ot_paper_train.py --model ./models/ot_paper_fused_fixed_bias_fused.pkl
"""
import os, sys, argparse, random, importlib.util
import numpy as np
import gym
import cloudpickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.custom_env  # noqa: F401
from stable_baselines3 import PPO

ENV_ID = "BipedalWalkerCustom-v0"


def _load_autodiff_callback():
    """AutoDifficultyCallback 住在 BipedalWalker-v3.py 裡，檔名有橫線不能直接 import。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BipedalWalker-v3.py")
    spec = importlib.util.spec_from_file_location("bw_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bw_main"] = mod
    spec.loader.exec_module(mod)
    return mod.AutoDifficultyCallback


def evaluate(model, n_episodes=10, difficulty=0.0, label=""):
    rewards = []
    for _ in range(n_episodes):
        e = gym.make(ENV_ID, difficulty=difficulty)
        obs = e.reset(seed=random.randint(0, 2**31 - 1))
        if isinstance(obs, tuple):
            obs = obs[0]
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            out = e.step(action)
            if len(out) == 5:
                obs, r, term, trunc, _ = out
                done = term or trunc
            else:
                obs, r, done, _ = out
            total += r
        rewards.append(total)
        e.close()
    m, s = float(np.mean(rewards)), float(np.std(rewards))
    print(f"  [{label}] mean reward = {m:.2f} +/- {s:.2f}")
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ot_paper_fuse.py 存出來的融合 policy (.pkl)")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--difficulty", type=float, default=0.0, help="訓練環境的起始難度")
    ap.add_argument("--n_eval", type=int, default=10)
    ap.add_argument("--out", default=None, help="預設在原檔名後面加 _trained")
    ap.add_argument("--no_auto_difficulty", action="store_true",
                    help="不掛 AutoDifficultyCallback（純粹固定難度訓練）")
    args = ap.parse_args()

    out_path = args.out or args.model.replace(".pkl", "_trained.pkl")

    env = gym.make(ENV_ID, difficulty=args.difficulty)
    print(f"載入融合模型：{args.model}")
    with open(args.model, "rb") as f:
        policy = cloudpickle.load(f)

    model = PPO("MlpPolicy", env, verbose=0, seed=1)
    model.policy = policy.to(model.device)

    print(f"\n===== 訓練前 =====")
    before = evaluate(model, args.n_eval, args.difficulty, "before")

    callbacks = []
    if not args.no_auto_difficulty:
        AutoDifficultyCallback = _load_autodiff_callback()
        callbacks.append(AutoDifficultyCallback(
            env, None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,
            shared_flags=None, cooldown_steps=0, hardseed_save_path="./logs/hard_seeds.json",
        ))
        print("已接上 AutoDifficultyCallback（難度自動升級）")

    print(f"\n===== 開始訓練，共 {args.steps} 步 =====")
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)

    print(f"\n===== 訓練後（難度 {args.difficulty}）=====")
    after = evaluate(model, args.n_eval, args.difficulty, "after")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.env = None
    with open(out_path, "wb") as f:
        cloudpickle.dump(model.policy, f)
    print(f"\n✅ 已儲存：{out_path}")

    print("\n" + "=" * 56)
    print(f"{os.path.basename(args.model)}")
    print(f"  訓練前  {before[0]:8.2f} +/- {before[1]:.2f}")
    print(f"  訓練後  {after[0]:8.2f} +/- {after[1]:.2f}   （{args.steps} 步）")
    print("=" * 56)
    env.close()


if __name__ == "__main__":
    main()
