# -*- coding: utf-8 -*-
"""
ot_progressive_iter 的「環境訓練版」：每一輪 OT 之後不做蒸餾，改在真實環境用 PPO 訓練。

設定（2026-10-04 指定）：
  - 4 層 × 4 輪；alpha 0.9 → 0.035（gamma = (0.035/0.9)^(1/3) ≈ 0.338）；ot_frac = 1
  - 每一輪 OT 後 PPO 10 萬步，難度固定 0.0，不接 AutoDifficultyCallback（2026-10-04 改）
  - 目的是讓模型產生擾動、讓 OT 逐輪重算得更準：開鏈式 T（上一層的 T 影響下一層）；
    各輪 PPO 只訓練交叉通道（凍結純通道，放開 value 輸出層）；每輪記錄 T 的變動與對齊誤差
    到 logs/ot_env_rounds/ot_T_log.json
  - 全部輪跑完後最終微調 100 萬步，難度重設為 0.5 再起跳
  - 訓練種子固定 0；hard seed 寫副本，不動主池
  - 方案 A（2026-10-04）：各輪之前先蒸餾一次當起點（同 distill_only：歸零後 5 epoch、lr 0.008、5% 資料）
  - 方案 C：起點、每輪 OT 寫入後、每輪訓練後各記一次走路分數（難度 0/0.3/0.6 各 10 局、固定地形）
輸出：logs/ot_env_rounds/ 下的 preppo（各輪訓練完、最終微調前）與 final 兩個模型，
接著用 eval_per_difficulty.py --seed 2026 評估。
"""
import os, sys, json, shutil
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import gym
import cloudpickle
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.custom_env  # noqa: F401
import tune_ot_progressive_iter as tune

bw = tune.bw
OUT_DIR = "./logs/ot_env_rounds"
N_ROUNDS, ALPHA_INIT, ALPHA_END, OT_FRAC = 4, 0.9, 0.035, 1.0
ROUND_STEPS, FINAL_STEPS, FINAL_START_DIFF = 100_000, 1_000_000, 0.5
SEED = 0
DISTILL_PT = "./logs/ga_eval/mtkd_continuous.pt(2)"
DISTILL_FRAC, DISTILL_EPOCHS, DISTILL_LR = 0.05, 5, 0.008
EVAL_DIFFS = [0.0, 0.3, 0.6]
EVAL_SEEDS = tune.make_seeds(4242, EVAL_DIFFS, 10)   # 跟搜尋/驗證/測試的地形都不同


def quick_eval(agent):
    return tune.evaluate(agent, EVAL_DIFFS, EVAL_SEEDS)[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hs_copy = os.path.join(OUT_DIR, "hard_seeds_copy.json")
    if not os.path.exists(hs_copy):
        src = "./logs/hard_seeds.json"
        if os.path.exists(src):
            shutil.copy(src, hs_copy)
        else:
            json.dump({}, open(hs_copy, "w"))

    torch.use_deterministic_algorithms(True, warn_only=True)
    tune.seed_everything(SEED)
    gamma = tune.gamma_from_end(ALPHA_INIT, ALPHA_END, N_ROUNDS)
    print(f"alpha {ALPHA_INIT} → {ALPHA_END}，gamma={gamma:.4f}，各輪 alpha："
          f"{[round(ALPHA_INIT * gamma ** r, 4) for r in range(N_ROUNDS)]}")

    env = gym.make("BipedalWalkerCustom-v0", difficulty=0.0)
    dad = tune.load_any("./best_model/model2.zip", env)
    mom = tune.load_any("./best_model/model3.zip", env)
    child = tune.build_child(dad, mom, env)

    print(f"讀取蒸餾資料：{DISTILL_PT}")
    full = torch.load(DISTILL_PT, map_location="cpu")
    blob = bw._subsample_distill_blob(full, DISTILL_FRAC)
    del full

    bw.progressive_iterative_ot_evolve(
        child, dad.policy, mom.policy, DISTILL_PT, env=env,
        distill_epochs=DISTILL_EPOCHS, distill_lr=DISTILL_LR, preloaded_blob=blob,
        initial_distill=True, round_eval_fn=quick_eval,
        n_rounds=N_ROUNDS, final_finetune_steps=FINAL_STEPS, ot_frac=OT_FRAC,
        device=str(child.device), alpha_init=ALPHA_INIT, alpha_gamma=gamma,
        pre_finetune_save_path=os.path.join(OUT_DIR, "preppo.pkl"),
        round_train="ppo", round_ppo_steps=ROUND_STEPS,
        hardseed_path=hs_copy, final_start_difficulty=FINAL_START_DIFF,
        round_auto_difficulty=False, round_difficulty=0.0,
        chain_T=True, round_freeze_pure=True,
        ot_log_path=os.path.join(OUT_DIR, "ot_T_log.json"),
    )
    with open(os.path.join(OUT_DIR, "final.pkl"), "wb") as f:
        cloudpickle.dump(child.policy, f)
    print(f"✅ 已存 {OUT_DIR}/preppo.pkl（最終微調前）與 {OUT_DIR}/final.pkl（最終）")


if __name__ == "__main__":
    main()
