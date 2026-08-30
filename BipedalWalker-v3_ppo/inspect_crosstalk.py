"""
inspect_crosstalk.py
比較父母原始交叉通道值 vs 融合後的值

用法：
  python inspect_crosstalk.py \
    --dad ./best_model/p_models/model3.zip \
    --mom ./best_model/p_models/model2.zip \
    --crossover ot_recursive
"""

import argparse
import copy
import sys
import os
import numpy as np
import torch
import torch.nn as nn

# 把主程式目錄加入 path，以便 import 主程式的函式
sys.path.insert(0, os.path.dirname(__file__))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dad", required=True)
    p.add_argument("--mom", required=True)
    p.add_argument("--crossover", default="zero",
                   choices=["zero", "ties", "ot", "ot_recursive"])
    p.add_argument("--ties_k", type=float, default=0.2)
    p.add_argument("--out", default="crosstalk_weights.npz",
                   help="輸出檔案路徑（.npz）")
    p.add_argument("--compare_alignment", action="store_true",
                   help="額外比較 weight-based vs activation-based 對齊的 Wasserstein 殘留誤差")
    p.add_argument("--env_id", default="BipedalWalkerCustom-v0")
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--n_obs_samples", type=int, default=512)
    return p.parse_args()


def load_model(path):
    from stable_baselines3 import PPO
    load_path = path[:-4] if path.endswith(".zip") else path
    return PPO.load(load_path, device="cpu")


def get_linear_layers(policy, skip=("action_net", "value_net")):
    return [
        (name, m) for name, m in policy.named_modules()
        if isinstance(m, nn.Linear) and not any(s in name for s in skip)
    ]



def compare_crosstalk(dad_policy, mom_policy, child_policy_raw, child_policy_fused, crossover, out_path):
    dad_mods  = dict(dad_policy.named_modules())
    mom_mods  = dict(mom_policy.named_modules())
    raw_mods  = dict(child_policy_raw.named_modules())
    fuse_mods = dict(child_policy_fused.named_modules())

    layers = get_linear_layers(child_policy_raw)
    arrays = {}  # 所有要存的陣列，key 為 "{層名}/{欄位}"

    for name, _ in layers:
        Wd     = dad_mods[name].weight.data.float().cpu().numpy()
        Wm     = mom_mods[name].weight.data.float().cpu().numpy()
        W_raw  = raw_mods[name].weight.data.float().cpu().numpy()
        W_fuse = fuse_mods[name].weight.data.float().cpu().numpy()

        oh, ih = Wd.shape[0] // 2, Wd.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        key = name.replace(".", "_")

        # 父母原始值
        arrays[f"{key}/dad_cross_top_right"]  = Wd[:oh, ih:]   # dad 上右（交叉區）
        arrays[f"{key}/mom_cross_bot_left"]   = Wm[oh:, :ih]   # mom 下左（交叉區）

        # raw child（未初始化）交叉通道
        arrays[f"{key}/raw_cross_top_right"]  = W_raw[:oh, ih:]
        arrays[f"{key}/raw_cross_bot_left"]   = W_raw[oh:, :ih]

        # 融合後交叉通道
        arrays[f"{key}/fused_X_md"]           = W_fuse[:oh, ih:]   # 上右：mom→dad
        arrays[f"{key}/fused_X_dm"]           = W_fuse[oh:, :ih]   # 下左：dad→mom

        # 純通道（供對照）
        arrays[f"{key}/dad_pure_top_left"]    = Wd[:oh, :ih]
        arrays[f"{key}/mom_pure_bot_right"]   = Wm[oh:, ih:]
        arrays[f"{key}/fused_dad_pure"]       = W_fuse[:oh, :ih]
        arrays[f"{key}/fused_mom_pure"]       = W_fuse[oh:, ih:]

    np.savez(out_path, **arrays)
    print(f"已儲存 {len(arrays)} 個陣列 → {out_path}.npz" if not out_path.endswith(".npz") else f"已儲存 {len(arrays)} 個陣列 → {out_path}")
    print("陣列 key 列表：")
    for k in arrays:
        print(f"  {k}  shape={arrays[k].shape}")


def compute_wasserstein_costs(dad_policy, mom_policy, bw, env=None, n_obs_samples=512):
    """
    對每一層算 weight-based 與（若給了 env）activation-based 的 OT 殘留誤差（Wasserstein cost）。
    cost 越小，代表 dad/mom 這層神經元用該種對齊方式找到的對應關係越接近「同一組神經元的排列」。
    """
    import ot as pot
    dad_mods = dict(dad_policy.named_modules())
    mom_mods = dict(mom_policy.named_modules())
    layer_names = [name for name, _ in get_linear_layers(dad_policy)]

    results = {}

    # --- weight-based ---
    for name in layer_names:
        Wd = dad_mods[name].weight.data
        Wm = mom_mods[name].weight.data
        oh, ih = Wd.shape[0] // 2, Wd.shape[1] // 2
        if oh == 0 or ih == 0:
            continue
        d = Wd[:oh, :ih].float().reshape(oh, -1).numpy().astype(np.float64)
        m_ = Wm[oh:, ih:].float().reshape(oh, -1).numpy().astype(np.float64)
        C = ((d[:, None, :] - m_[None, :, :]) ** 2).sum(-1)
        mu = np.ones(oh) / oh
        nu = np.ones(oh) / oh
        cost_w = pot.emd2(mu, nu, C)
        results.setdefault(name, {})["weight_based_cost"] = float(cost_w)
        results[name]["n_neurons"] = oh

    # --- activation-based（需要 env）---
    if env is not None:
        print(f"  蒐集 {n_obs_samples} 筆真實觀測中...")
        obs_batch = bw._sample_observations(env, n_samples=n_obs_samples)
        dad_acts = bw._collect_pure_activations(dad_policy, obs_batch, layer_names)
        mom_acts = bw._collect_pure_activations(mom_policy, obs_batch, layer_names)
        for name in layer_names:
            Wd = dad_mods[name].weight.data
            oh, ih = Wd.shape[0] // 2, Wd.shape[1] // 2
            if oh == 0 or ih == 0:
                continue
            da = dad_acts[name][:, :oh].t().numpy().astype(np.float64)  # (n, m)
            ma = mom_acts[name][:, oh:].t().numpy().astype(np.float64)  # (n, m)
            C = ((da[:, None, :] - ma[None, :, :]) ** 2).sum(-1)
            mu = np.ones(oh) / oh
            nu = np.ones(oh) / oh
            cost_a = pot.emd2(mu, nu, C)
            results.setdefault(name, {})["activation_based_cost"] = float(cost_a)

    return results


def print_wasserstein_report(results):
    print(f"\n{'='*70}")
    print("  Wasserstein 殘留誤差比較（越小代表 dad/mom 神經元對齊得越好）")
    print(f"{'='*70}")
    header = f"  {'層名':40s} {'神經元數':>8s} {'weight-based':>14s} {'activation-based':>16s} {'改善%':>8s}"
    print(header)
    for name, r in results.items():
        n = r.get("n_neurons", "-")
        cw = r.get("weight_based_cost")
        ca = r.get("activation_based_cost")
        cw_s = f"{cw:.4f}" if cw is not None else "-"
        ca_s = f"{ca:.4f}" if ca is not None else "-"
        if cw is not None and ca is not None and cw > 0:
            improve = (cw - ca) / cw * 100
            improve_s = f"{improve:+.1f}%"
        else:
            improve_s = "-"
        print(f"  {name:40s} {str(n):>8s} {cw_s:>14s} {ca_s:>16s} {improve_s:>8s}")
    print(f"{'='*70}\n")


def main():
    args = parse_args()

    print(f"載入 dad：{args.dad}")
    dad = load_model(args.dad)
    print(f"載入 mom：{args.mom}")
    mom = load_model(args.mom)

    # 從主程式 import 函式（檔名含連字符，需用 importlib 載入）
    import importlib.util
    main_py = os.path.join(os.path.dirname(__file__), "BipedalWalker-v3.py")
    spec = importlib.util.spec_from_file_location("bw_main", main_py)
    bw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bw)

    create_dual_channel_policy      = bw.create_dual_channel_policy
    zero_initialize_crosstalk       = bw.zero_initialize_crosstalk
    ties_initialize_crosstalk       = bw.ties_initialize_crosstalk
    ot_initialize_crosstalk         = bw.ot_initialize_crosstalk
    recursive_ot_initialize_crosstalk = bw.recursive_ot_initialize_crosstalk

    env = None
    if args.crossover == "ot_recursive" or args.compare_alignment:
        import gym
        print(f"建立環境：{args.env_id}（difficulty={args.difficulty}）")
        env = gym.make(args.env_id, difficulty=args.difficulty)

    if args.compare_alignment:
        print("\n正在比較 weight-based vs activation-based 對齊誤差...")
        results = compute_wasserstein_costs(
            dad.policy, mom.policy, bw, env=env, n_obs_samples=args.n_obs_samples
        )
        print_wasserstein_report(results)

    # 建立 raw child（未初始化）
    child_raw = create_dual_channel_policy(dad.policy, mom.policy)

    # 建立 fused child
    child_fused = copy.deepcopy(child_raw)
    if args.crossover == "ties":
        ties_initialize_crosstalk(child_fused, dad.policy, mom.policy, k=args.ties_k)
    elif args.crossover == "ot":
        ot_initialize_crosstalk(child_fused, dad.policy, mom.policy)
    elif args.crossover == "ot_recursive":
        recursive_ot_initialize_crosstalk(child_fused, dad.policy, mom.policy, env=env)
    else:
        zero_initialize_crosstalk(child_fused)

    compare_crosstalk(dad.policy, mom.policy, child_raw, child_fused, args.crossover, args.out)


if __name__ == "__main__":
    main()
