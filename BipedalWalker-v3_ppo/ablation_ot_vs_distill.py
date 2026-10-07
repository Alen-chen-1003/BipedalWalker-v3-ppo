# -*- coding: utf-8 -*-
"""
OT 到底有沒有貢獻？用跟 Optuna trial 35 同一批 5% 蒸餾資料做兩個對照組：

  ① t35_no_ot：參數完全照 trial 35（3 輪、7 epoch、lr、ot_frac 都一樣），只把
     alpha_init 設 0——_iterative_refine_single_layer 在 alpha=0 時交叉通道原值不動，
     等於同樣的流程、同樣的蒸餾量，只是沒有 OT。交叉通道起點是 create_dual_channel_policy
     的直接拼接值（跟 trial 35 的起點相同）。
  ② distill_only_5pct：跟 --crossover distill_only 相同（交叉通道先歸零，蒸餾 5 epoch、
     lr=0.008），只是資料換成同一批 5%。

兩個模型存成 PPO 前的 policy，接著用 eval_per_difficulty.py --seed 2026 跟 trial 35
及原本的 distill_only 配對比較。

  ③ chain_layer：trial 35 參數，改成「只蒸餾這一層」+「上一層的 T 影響下一層」（鏈式）。
  ④ chain_layer_no_ot：同 ③ 但 alpha_init=0，新流程下的無 OT 對照組。

用法：
  python ablation_ot_vs_distill.py                                   # ①②
  python ablation_ot_vs_distill.py --which chain_layer chain_layer_no_ot   # ③④
"""
import os, sys, argparse
import gym
import cloudpickle
import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.custom_env  # noqa: F401

import tune_ot_progressive_iter as tune  # 共用 load_any / build_child，以及已載入的 bw 主程式

bw = tune.bw
ENV_ID = "BipedalWalkerCustom-v0"
OUT_DIR = "./logs/optuna_ot_prog_iter"
DISTILL_PT = "./logs/ga_eval/mtkd_continuous.pt(2)"
DISTILL_FRAC = 0.05   # 跟調參時相同，_subsample_distill_blob 固定 seed，抽到的是同一批


def run_prog_iter(p, dad, mom, env, blob, out_name, alpha_init, gamma, scope, chain):
    child = tune.build_child(dad, mom, env)
    bw.progressive_iterative_ot_evolve(
        child, dad.policy, mom.policy, DISTILL_PT, env=env,
        n_rounds=p["n_rounds"], distill_epochs=p["distill_epochs"], distill_lr=p["distill_lr"],
        final_finetune_steps=0, ot_frac=p["ot_frac"], device=str(child.device),
        alpha_init=alpha_init, alpha_gamma=gamma, pre_finetune_save_path=None,
        preloaded_blob=blob, distill_scope=scope, chain_T=chain,
    )
    with open(f"{OUT_DIR}/{out_name}.pkl", "wb") as f:
        cloudpickle.dump(child.policy, f)
    print(f"✅ 已存 {OUT_DIR}/{out_name}.pkl")


VARIANTS = ["t35_no_ot", "distill_only_5pct", "chain_layer", "chain_layer_no_ot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["t35_no_ot", "distill_only_5pct"], choices=VARIANTS,
                    help="要跑哪些對照組（預設＝最初那兩個）。chain_layer＝trial 35 參數 + 只蒸餾這一層 + 鏈式 T；"
                         "chain_layer_no_ot＝同上但 alpha_init=0（不做 OT），用來單獨看 OT 在新流程裡的貢獻")
    args = ap.parse_args()

    study = optuna.load_study(study_name="ot_prog_iter", storage=f"sqlite:///{OUT_DIR}/study.db")
    t = study.trials[35]
    p = t.params
    gamma = t.user_attrs["alpha_gamma"]
    print(f"trial 35 參數：{p}  gamma={gamma:.4f}")

    env = gym.make(ENV_ID, difficulty=0.0)
    dad = tune.load_any("./best_model/model2.zip", env)
    mom = tune.load_any("./best_model/model3.zip", env)

    print(f"讀取蒸餾資料：{DISTILL_PT}")
    full = torch.load(DISTILL_PT, map_location="cpu")
    blob = bw._subsample_distill_blob(full, DISTILL_FRAC)
    del full

    if "t35_no_ot" in args.which:
        # ① 同 trial 35 流程，關掉 OT
        print("\n===== ① t35_no_ot：trial 35 參數，alpha_init=0（不做 OT）=====")
        run_prog_iter(p, dad, mom, env, blob, "ablation_t35_no_ot", 0.0, 1.0, "all", False)

    if "distill_only_5pct" in args.which:
        # ② distill_only，同一批 5% 資料
        print("\n===== ② distill_only_5pct：歸零後蒸餾 5 epoch，5% 資料 =====")
        child = tune.build_child(dad, mom, env)
        bw.distill_crosstalk_baseline(
            child, DISTILL_PT, epochs=5, lr=0.008, device=str(child.device),
            zero_init=True, preloaded_blob=blob,
        )
        with open(f"{OUT_DIR}/ablation_distill_only_5pct.pkl", "wb") as f:
            cloudpickle.dump(child.policy, f)

    if "chain_layer" in args.which:
        print("\n===== ③ chain_layer：trial 35 參數 + 只蒸餾這一層 + 鏈式 T =====")
        run_prog_iter(p, dad, mom, env, blob, "ablation_chain_layer", p["alpha_init"], gamma, "layer", True)

    if "chain_layer_no_ot" in args.which:
        print("\n===== ④ chain_layer_no_ot：同 ③ 但 alpha_init=0（不做 OT）=====")
        run_prog_iter(p, dad, mom, env, blob, "ablation_chain_layer_no_ot", 0.0, 1.0, "layer", True)

    print(f"\n✅ 完成：{args.which}")


if __name__ == "__main__":
    main()
