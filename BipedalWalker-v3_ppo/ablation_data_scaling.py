# -*- coding: utf-8 -*-
"""
OT 能不能讓融合需要的蒸餾資料變少？

用 inter_ot（v2 調參 trial #18）的參數，把蒸餾資料一路降到 1% / 0.2% / 0.06%，
每個資料量各訓練「有 OT」和「alpha_init=0（不做 OT）」兩個模型；另外存一個完全不訓練的
直接拼接模型當下限。若資料很少時「沒有 OT」掉分、「有 OT」還撐得住，就表示 OT 讓
融合比較省資料。

- 抽樣用 _subsample_distill_blob（固定 seed 0 的同一個亂數排列取前 k 筆），所以小的
  資料量是大的資料量的子集，各資料量之間只差在筆數。
- 每輪蒸餾的 epoch 數照 #18（7 epoch），資料越少梯度步數越少；同一個資料量下，有 OT
  和沒有 OT 的步數完全相同。
- 最小只到 0.06%：蒸餾 batch_size=4096 且 drop_last，0.05%（約 3600 筆）湊不滿一個 batch。

用法：
  python ablation_data_scaling.py
接著：
  python eval_per_difficulty.py --seed 2026 --only <下面 DEFAULTS 裡的 scale_* 與 raw_concat>
"""
import os, sys
import gym
import cloudpickle
import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.custom_env  # noqa: F401

import tune_ot_progressive_iter as tune

bw = tune.bw
ENV_ID = "BipedalWalkerCustom-v0"
OUT_DIR = "./logs/data_scaling"
DISTILL_PT = "./logs/ga_eval/mtkd_continuous.pt(2)"
FRACS = [0.01, 0.002, 0.0006]


def frac_tag(f):
    return f"{f * 100:g}pct".replace(".", "p")  # 0.01 → 1pct、0.002 → 0p2pct、0.0006 → 0p06pct


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    study = optuna.load_study(study_name="ot_prog_iter_v2",
                              storage="sqlite:///./logs/optuna_ot_prog_iter_v2/study.db")
    t = study.trials[18]
    p, gamma = t.params, t.user_attrs["alpha_gamma"]
    print(f"inter_ot（v2 #18）參數：{p}  gamma={gamma:.4f}")

    env = gym.make(ENV_ID, difficulty=0.0)
    dad = tune.load_any("./best_model/model2.zip", env)
    mom = tune.load_any("./best_model/model3.zip", env)

    # 下限：直接拼接，完全不訓練
    child = tune.build_child(dad, mom, env)
    with open(f"{OUT_DIR}/raw_concat.pkl", "wb") as f:
        cloudpickle.dump(child.policy, f)
    print("✅ 已存 raw_concat（直接拼接，不做 OT、不蒸餾）")

    print(f"讀取蒸餾資料：{DISTILL_PT}")
    full = torch.load(DISTILL_PT, map_location="cpu")

    for frac in FRACS:
        blob = bw._subsample_distill_blob(full, frac)
        for use_ot in (True, False):
            name = f"scale_{frac_tag(frac)}_{'ot' if use_ot else 'no_ot'}"
            print(f"\n===== {name}（資料 {frac * 100:g}%，{'有' if use_ot else '沒有'} OT）=====")
            child = tune.build_child(dad, mom, env)
            bw.progressive_iterative_ot_evolve(
                child, dad.policy, mom.policy, DISTILL_PT, env=env,
                n_rounds=p["n_rounds"], distill_epochs=p["distill_epochs"], distill_lr=p["distill_lr"],
                final_finetune_steps=0, ot_frac=p["ot_frac"], device=str(child.device),
                alpha_init=p["alpha_init"] if use_ot else 0.0, alpha_gamma=gamma if use_ot else 1.0,
                pre_finetune_save_path=None, preloaded_blob=blob,
            )
            with open(f"{OUT_DIR}/{name}.pkl", "wb") as f:
                cloudpickle.dump(child.policy, f)
            print(f"✅ 已存 {OUT_DIR}/{name}.pkl")

        # distill_only：跟 --crossover distill_only 相同（交叉通道先歸零，蒸餾一次 5 epoch、lr 0.008），
        # 只是資料換成同一個資料量
        name = f"scale_{frac_tag(frac)}_distill_only"
        print(f"\n===== {name}（資料 {frac * 100:g}%，歸零後蒸餾 5 epoch，沒有 OT）=====")
        child = tune.build_child(dad, mom, env)
        bw.distill_crosstalk_baseline(
            child, DISTILL_PT, epochs=5, lr=0.008, device=str(child.device),
            zero_init=True, preloaded_blob=blob,
        )
        with open(f"{OUT_DIR}/{name}.pkl", "wb") as f:
            cloudpickle.dump(child.policy, f)
        print(f"✅ 已存 {OUT_DIR}/{name}.pkl")
        del blob

    print("\n✅ 全部完成")


if __name__ == "__main__":
    main()
