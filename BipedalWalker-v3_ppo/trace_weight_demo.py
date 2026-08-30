"""
用真實的 dad/mom 模型，追蹤「progressive_iterative_ot_evolve」第一層第一格交叉通道權重
在每一步(OT 對齊 / 蒸餾消化)之後的真實數值——不是模擬，是實際呼叫程式碼跑出來的。

用法：
    python trace_weight_demo.py --dad ./best_model/p_models/model3.zip \
        --mom ./best_model/p_models/model2.zip \
        --distill_pt "./logs/ga_eval/mtkd_continuous.pt(2)" \
        --n_rounds 5 --distill_epochs 5
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import gym
import torch
from stable_baselines3 import PPO


def load_bw3_module():
    """BipedalWalker-v3.py 檔名有連字號，不能直接 import，用 importlib 動態載入。"""
    script_path = Path(__file__).parent / "BipedalWalker-v3.py"
    spec = importlib.util.spec_from_file_location("bw3", script_path)
    bw3 = importlib.util.module_from_spec(spec)
    sys.modules["bw3"] = bw3
    spec.loader.exec_module(bw3)
    return bw3


def main():
    parser = argparse.ArgumentParser(description="追蹤一格交叉通道權重在 OT+蒸餾迴圈中的真實變化")
    parser.add_argument("--dad", type=str, required=True)
    parser.add_argument("--mom", type=str, required=True)
    parser.add_argument("--distill_pt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--distill_epochs", type=int, default=5)
    parser.add_argument("--distill_lr", type=float, default=0.008)
    parser.add_argument("--stage", type=int, default=0, help="要追蹤哪一層 (0=第一個可融合層)")
    args = parser.parse_args()

    print("載入 BipedalWalker-v3.py 中的函式...")
    bw3 = load_bw3_module()

    print("import env.custom_env（註冊 BipedalWalkerCustom-v0）...")
    import env.custom_env  # noqa: F401

    env = gym.make("BipedalWalkerCustom-v0", difficulty=0.0, render_mode=None)

    print(f"載入 dad：{args.dad}")
    dad_path = args.dad[:-4] if args.dad.endswith(".zip") else args.dad
    dad = PPO.load(dad_path, env=env, device="cpu")
    print(f"載入 mom：{args.mom}")
    mom_path = args.mom[:-4] if args.mom.endswith(".zip") else args.mom
    mom = PPO.load(mom_path, env=env, device="cpu")

    print("建構 child（create_dual_channel_policy）...")
    child_policy = bw3.create_dual_channel_policy(dad.policy, mom.policy)
    child_agent = PPO("MlpPolicy", env, verbose=0)
    child_agent.policy = child_policy.to(child_agent.device)
    child_agent.policy.optimizer = torch.optim.Adam(child_agent.policy.parameters(), lr=1e-4)

    layer_names = bw3._fusable_layer_names(child_policy)
    layer_name = layer_names[args.stage]
    print(f"可融合層：{layer_names}，這次追蹤第 {args.stage} 層：{layer_name}\n")

    named = dict(child_policy.named_modules())
    W = named[layer_name].weight.data
    oh, ih = W.shape[0] // 2, W.shape[1] // 2
    # 追蹤的目標格：交叉通道區塊 W[:oh, ih:] 的左上角那一格
    r, c = 0, ih

    def snapshot(label):
        val = W[r, c].item()
        print(f"  [{label}] W[{r},{c}] = {val:.6f}")
        return val

    print("===== 拼接完成，尚未做任何 OT/蒸餾 =====")
    trace = [("拼接繼承值", snapshot("拼接繼承值"))]

    print(f"\n===== 蒸餾資料只讀一次：{args.distill_pt} =====")
    blob = torch.load(args.distill_pt, map_location="cpu")

    for round_idx in range(args.n_rounds):
        print(f"\n----- Round {round_idx + 1}/{args.n_rounds} -----")
        if round_idx == 0:
            bw3.recursive_ot_fuse_single_layer(
                child_policy, dad.policy, mom.policy, args.stage,
                align_first_layer=True, average_with_parent=True,
            )
            trace.append((f"Round {round_idx+1} OT對齊後", snapshot("OT 對齊後")))
        else:
            bw3._iterative_refine_single_layer(child_policy, dad.policy, mom.policy, args.stage)
            trace.append((f"Round {round_idx+1} 重新配對後", snapshot("重新配對後")))

        bw3.distill_crosstalk_baseline(
            child_agent, args.distill_pt, epochs=args.distill_epochs, lr=args.distill_lr,
            device=args.device, zero_init=False, preloaded_blob=blob,
        )
        trace.append((f"Round {round_idx+1} 蒸餾消化後", snapshot("蒸餾消化後")))

    print("\n" + "=" * 50)
    print(f"  追蹤層：{layer_name}  座標：W[{r},{c}]")
    print("-" * 50)
    for label, val in trace:
        print(f"  {label:<20s} {val:.6f}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
