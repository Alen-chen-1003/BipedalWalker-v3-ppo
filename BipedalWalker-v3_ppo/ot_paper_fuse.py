# -*- coding: utf-8 -*-
"""
照 arXiv:2207.00978 (Renaissance Robot) 官方 cheetah.ipynb 的方式融合兩個模型
（--method paper，預設），也可以改用本專案 --crossover ot_iter 的迭代 OT
雙通道拼接法（--method ot_iter），方便同一組 agent1/agent2 直接比較兩種融合
方式的效果。

兩種方法的差異：
  paper   （align_nodes）：兩個網路的神經元一對一配對後直接平均，產生一個
           「跟單一 parent 同寬」的網路，每顆神經元都是 dad/mom 的混血。
  ot_iter （splice_dual_channel + iterative_ot_splice）：每層輸出神經元對半
           拆——上半沿用 dad 整列權重、下半沿用 mom 整列權重（神經元身分不
           變），再用 OT 疊代修正兩個 cross-talk 區塊；跟本專案 --crossover
           ot_iter（BipedalWalker-v3.py 的 iterative_ot_initialize_crosstalk）
           是同一套邏輯，只是搬到這裡的 numpy 權重流程上運作。

用法：
  python ot_paper_fuse.py --agent1 ./best_model/p_models/model2.zip \
                          --agent2 ./best_model/p_models/model3.zip \
                          --method both
"""
import os, sys, argparse, random, copy
import numpy as np
import torch as pt
import gym
import cloudpickle
import ot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.custom_env  # noqa: F401  註冊 BipedalWalkerCustom-v0
from stable_baselines3 import PPO

ENV_ID = "BipedalWalkerCustom-v0"

W_KEYS = ["mlp_extractor.policy_net.0.weight",
          "mlp_extractor.policy_net.2.weight",
          "action_net.weight"]
B_KEYS = ["mlp_extractor.policy_net.0.bias",
          "mlp_extractor.policy_net.2.bias",
          "action_net.bias"]


# ── 取權重（對應論文的 get_weights）────────────────────────────────────────
def get_weights(model):
    p = model.get_parameters()["policy"]
    return [p[k].detach().cpu().numpy().copy() for k in W_KEYS]


def get_biases(model):
    p = model.get_parameters()["policy"]
    return [p[k].detach().cpu().numpy().copy() for k in B_KEYS]


# ── 論文的 align_nodes，另外把排列本身也回傳 ──────────────────────────────
def align_nodes(layer_weights_all, stop_layer_k=3, permute_last=True,
                col_dir="paper", verbose=True):
    """
    忠實移植論文的 align_nodes，額外做三件事：
      1. 把每一層算出的排列一起回傳（論文只在迴圈內當區域變數用掉）
      2. 排列索引除了照論文用浮點數精確比對反推，也用耦合矩陣算一份對照，
         兩者不一致時出聲——論文那個 np.argwhere(aligned_Xs == Xs[w])[0][0]
         比對的是「任一元素相等」而不是「整列相等」，理論上可能抓錯列。
      3. col_dir 可以切換下一層欄位重排的方向。

    col_dir：
      "paper" —— 照論文寫的 w[alignment_idx]（gather perm）
      "fixed" —— w[argsort(alignment_idx)]

    ⚠️ 實測結果：論文那個方向是反的。inverse_transform 產生的列順序等於
    Xs[argsort(perm)]，所以下一層的欄位也必須用 argsort(perm) 重排才對得起來。
    用 perm 去 gather 會把上一層的神經元對應關係打亂——實測對單一模型套用
    論文方向後，動作輸出最大變化 13.16（完全變成另一個網路）；用 argsort(perm)
    則是 7.2e-07（只有 float32 誤差）。也就是說論文在對齊第 2、3 層時，
    餵進 OT 的 source 權重欄位已經是亂的。

    permute_last=False 時，最後一層（action_net）只重排欄位、不重排列。
    輸出維度是各關節扭矩、有固定物理語意，把列打亂等於把動作接錯關節。
    """
    assert col_dir in ("paper", "fixed")
    perms = []
    alignment_idx = None

    for i in range(stop_layer_k):
        for j in range(1, len(layer_weights_all)):
            Xs = layer_weights_all[j][i].copy()
            Xt = layer_weights_all[0][i]

            # 論文的 `if i > 0`：用上一層的排列重排這一層的欄位
            if i > 0 and alignment_idx is not None:
                col_idx = alignment_idx if col_dir == "paper" else np.argsort(alignment_idx)
                Xs = Xs[:, col_idx]

            is_last = (i == stop_layer_k - 1)
            if is_last and not permute_last:
                layer_weights_all[j][i] = Xs
                alignment_idx = np.arange(Xs.shape[0])
                perms.append(alignment_idx)
                if verbose:
                    print(f"  層 {i}: shape={Xs.shape} 只重排欄位，不重排列（保留動作維度語意）")
                continue

            ot_emd = ot.da.EMDTransport()
            ot_emd.fit(Xs=Xs, Xt=Xt)
            aligned_Xs = ot_emd.inverse_transform(Xt=Xt)
            layer_weights_all[j][i] = aligned_Xs

            # (a) 論文的反推方式
            paper_idx = []
            for w in range(Xs.shape[0]):
                hit = np.argwhere(aligned_Xs == Xs[w])
                paper_idx.append(int(hit[0][0]) if len(hit) else -1)
            paper_idx = np.array(paper_idx)

            # (b) 直接從耦合矩陣取（coupling 是 (n_s, n_t)，第 k 列的最大值位置就是 k 的去處）
            coup_idx = np.asarray(ot_emd.coupling_).argmax(axis=1)

            if not np.array_equal(paper_idx, coup_idx):
                print(f"  ⚠️ 層 {i}: 論文的精確比對反推 與 耦合矩陣反推 不一致，改用耦合矩陣")
                print(f"     paper={paper_idx[:12]}...\n     coup ={coup_idx[:12]}...")
            alignment_idx = coup_idx
            perms.append(alignment_idx)

            n_uniq = len(np.unique(alignment_idx))
            if verbose:
                print(f"  層 {i}: shape={Xs.shape} 配對完成，"
                      f"排列涵蓋 {n_uniq}/{len(alignment_idx)} 個不同位置"
                      f"{'（是合法排列）' if n_uniq == len(alignment_idx) else ' ⚠️ 有重複，不是合法排列'}")

    return layer_weights_all, perms


# ── 平均（對應論文的 fuse_policy_weights）─────────────────────────────────
def fuse(aligned_all):
    return [(aligned_all[0][i] + aligned_all[1][i]) / 2.0 for i in range(len(aligned_all[0]))]


# ── 移植自 BipedalWalker-v3.py 的 ot_iter（迭代 OT 雙通道拼接）──────────────
# 跟上面論文的 align_nodes 完全不同路數：
#   align_nodes：兩個網路的神經元「一對一配對後直接平均」，輸出網路跟單一
#                parent 同寬，每顆神經元都是 dad/mom 的混血。
#   ot_iter    ：先把每層輸出神經元「對半拆」——上半神經元直接沿用 dad 的整列
#                權重、下半沿用 mom 的整列權重（神經元本身不混血、身分不變），
#                再用 OT 疊代修正「dad 半邊神經元去讀 mom 半邊輸入」與
#                「mom 半邊神經元去讀 dad 半邊輸入」這兩個 cross-talk 區塊——
#                因為整列複製後，dad 的列在原本的網路裡只認得 dad 自己那一半
#                輸入，套到混合後一半是 mom 神經元的輸入上，數值是對不起來的，
#                需要重新用 OT 算。
# 輸出層（action_net）跟 align_nodes 的 permute_last=False 邏輯一致：整層
# 固定用 dad 權重，不拆分——動作輸出有固定物理語意，拆一半等於把扭矩接錯關節。

def splice_dual_channel(W1, W2, B1=None, B2=None):
    """對半拆拼接（對應 BipedalWalker-v3.py 的 create_dual_channel_policy）：
    隱藏層（W_KEYS 除最後一層外）輸出神經元上半沿用 W1（dad）、下半沿用
    W2（mom）；最後一層（action_net）整層固定用 W1，不拆分。"""
    n_layers = len(W1)
    out_W, out_B = [], []
    for i in range(n_layers):
        w1 = W1[i]
        if i == n_layers - 1:
            out_W.append(w1.copy())
            if B1 is not None:
                out_B.append(B1[i].copy())
            continue
        w = w1.copy()
        half = w1.shape[0] // 2
        w[half:, :] = W2[i][half:, :]
        out_W.append(w)
        if B1 is not None:
            b = B1[i].copy()
            b[half:] = B2[i][half:]
            out_B.append(b)
    return out_W, (out_B if B1 is not None else None)


def _compute_layer_transport(dad_W, mom_W, eps=1e-7):
    """移植自 BipedalWalker-v3.py 的 _compute_layer_transport：算 dad/mom 純
    通道權重之間的 OT 傳輸矩陣 T，column-normalize 成「軟置換」。"""
    n = dad_W.shape[0]
    d = dad_W.reshape(n, -1).astype(np.float64)
    m = mom_W.reshape(n, -1).astype(np.float64)
    diff = d[:, None, :] - m[None, :, :]
    C = (diff ** 2).sum(axis=-1)
    mu = np.ones(n) / n
    nu = np.ones(n) / n
    T = ot.emd(mu, nu, C)
    col_sums = T.sum(axis=0)
    col_sums[col_sums < eps] = eps
    return T / col_sums[None, :]


def iterative_ot_splice(W1, W2, B1, B2, n_iters=5, ot_frac=1.0, verbose=True):
    """移植自 BipedalWalker-v3.py 的 iterative_ot_initialize_crosstalk，改寫成
    在 ot_paper_fuse.py 的 numpy 權重清單（W_KEYS 順序）上運作，好跟論文那條
    align_nodes 路徑共用同一套 evaluate/build_fused/存檔流程。"""
    n_layers = len(W1)
    hidden = list(range(n_layers - 1))  # 排除最後一層 action_net
    W, _ = splice_dual_channel(W1, W2)  # 起始：naive 整列拼接（cross-talk 區塊還沒修正）

    for it in range(n_iters):
        T_prev = None
        for li in hidden:
            w = W[li]
            oh, ih = w.shape[0] // 2, w.shape[1] // 2
            if oh == 0 or ih == 0:
                continue
            wd_pure = W1[li][:oh, :ih]
            wm_pure = W2[li][oh:, ih:]
            if it == 0:
                if T_prev is None:
                    w[:oh, ih:] = wd_pure.copy()
                    w[oh:, :ih] = wm_pure.copy()
                else:
                    w[:oh, ih:] = wd_pure @ T_prev
                    w[oh:, :ih] = wm_pure @ T_prev.T
                T_prev = _compute_layer_transport(wd_pure, wm_pure)
            else:
                x_md_prev = w[:oh, ih:].copy()
                x_dm_prev = w[oh:, :ih].copy()
                T_md = _compute_layer_transport(x_md_prev.T.copy(), wd_pure.T.copy())
                T_dm = _compute_layer_transport(x_dm_prev.T.copy(), wm_pure.T.copy())
                w[:oh, ih:] = x_md_prev @ T_md
                w[oh:, :ih] = x_dm_prev @ T_dm
                T_prev = T_md
        if verbose:
            xmd = np.concatenate([
                W[li][:W[li].shape[0] // 2, W[li].shape[1] // 2:].flatten() for li in hidden
            ])
            print(f"  [ot_iter] iter {it + 1}/{n_iters}  X_md norm={np.linalg.norm(xmd):.4f}")

    # 疊代完成後跟父母純通道做 0.5 平均（防止多次 @T 數值收縮，同時保留對齊結果）
    for li in hidden:
        w = W[li]
        oh, ih = w.shape[0] // 2, w.shape[1] // 2
        if oh == 0 or ih == 0:
            continue
        wd_pure = W1[li][:oh, :ih]
        wm_pure = W2[li][oh:, ih:]
        x_md_final = 0.5 * (w[:oh, ih:] + wd_pure)
        x_dm_final = 0.5 * (w[oh:, :ih] + wm_pure)
        k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
        if k < oh:
            print(f"  [ot_iter] 層 {li}：只保留前 {k}/{oh} 列為疊代結果，其餘歸零")
            w[:oh, ih:] = np.random.normal(0.0, 0.001, size=w[:oh, ih:].shape)
            w[oh:, :ih] = np.random.normal(0.0, 0.001, size=w[oh:, :ih].shape)
        if k > 0:
            w[:k, ih:] = x_md_final[:k]
            w[oh:oh + k, :ih] = x_dm_final[:k]

    _, spliced_B = splice_dual_channel(W1, W2, B1, B2)
    return W, spliced_B


# ── 檢驗排列方向是否正確 ──────────────────────────────────────────────────
def check_permutation_consistency(model, perms, obs_batch, col_dir="paper"):
    """
    對「單一模型」套用排列：第 i 層的列照 inverse_transform 實際的順序
    （= Xs[argsort(perm)]），第 i+1 層的欄照 col_dir 指定的方向。

    如果兩者方向一致，這只是把神經元換位置，網路輸出必須完全不變；
    輸出變了就代表欄位重排跟列重排對不起來，深層拿到的是被打亂的輸入。

    注意不能對列和欄用同一個索引陣列去測——那樣是恆等變換、必然通過，
    測不到論文實際的用法。
    """
    pol = copy.deepcopy(model.policy)
    sd = pol.state_dict()
    W = [sd[k].clone() for k in W_KEYS]
    B = [sd[k].clone() for k in B_KEYS]

    for i in range(2):                      # 只重排兩層隱藏層的神經元
        p = np.asarray(perms[i])
        row_idx = pt.as_tensor(np.argsort(p), dtype=pt.long)   # 列：inverse_transform 的實際順序
        col_idx = pt.as_tensor(p if col_dir == "paper" else np.argsort(p), dtype=pt.long)
        W[i] = W[i][row_idx]
        B[i] = B[i][row_idx]
        W[i + 1] = W[i + 1][:, col_idx]

    for k, v in zip(W_KEYS, W):
        sd[k] = v
    for k, v in zip(B_KEYS, B):
        sd[k] = v
    pol.load_state_dict(sd)

    with pt.no_grad():
        a0 = model.policy(obs_batch, deterministic=True)[0]
        a1 = pol(obs_batch, deterministic=True)[0]
    return (a0 - a1).abs().max().item()


# ── 評估 ──────────────────────────────────────────────────────────────────
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


def load_model(path, env):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            policy = cloudpickle.load(f)
        m = PPO("MlpPolicy", env, verbose=0, seed=1)
        m.policy = policy.to(m.device)
        return m
    return PPO.load(path[:-4] if path.endswith(".zip") else path, env=env, device="cpu")


def build_fused(env, fused_W, fused_B=None):
    """對應論文的 replace_policy：新建一個 PPO，只覆寫權重（可選擇也覆寫 bias）。"""
    model = PPO("MlpPolicy", env, verbose=0, seed=1)
    params = model.get_parameters()
    pol = params["policy"]
    for k, w in zip(W_KEYS, fused_W):
        pol[k] = pt.from_numpy(np.ascontiguousarray(w)).float()
    if fused_B is not None:
        for k, b in zip(B_KEYS, fused_B):
            pol[k] = pt.from_numpy(np.ascontiguousarray(b)).float()
    model.set_parameters({"policy": pol}, exact_match=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent1", default="./best_model/p_models/model2.zip",
                    help="論文的 target，全程固定不動")
    ap.add_argument("--agent2", default="./best_model/p_models/model3.zip",
                    help="論文的 source，會被重排對齊到 agent1")
    ap.add_argument("--n_eval", type=int, default=10)
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--out_dir", default="./models")
    ap.add_argument("--method", default="paper", choices=["paper", "ot_iter", "both"],
                    help="paper：論文的 align_nodes 一對一配對+平均（預設，行為不變）；"
                         "ot_iter：本專案 --crossover ot_iter 的迭代 OT 雙通道拼接；"
                         "both：兩種都跑，方便直接比較同一組 agent1/agent2")
    ap.add_argument("--ot_iters", type=int, default=5, help="ot_iter 方法的疊代輪數")
    ap.add_argument("--ot_frac", type=float, default=1.0,
                    help="ot_iter 方法：cross-talk 區塊只保留前 ot_frac 比例的列為疊代結果，"
                         "其餘歸零小噪音（1.0 = 整塊都用疊代結果）")
    args = ap.parse_args()

    env = gym.make(ENV_ID, difficulty=args.difficulty)
    print(f"載入 agent1 (target)：{args.agent1}")
    m1 = load_model(args.agent1, env)
    print(f"載入 agent2 (source)：{args.agent2}")
    m2 = load_model(args.agent2, env)

    W1, W2 = get_weights(m1), get_weights(m2)
    B1, B2 = get_biases(m1), get_biases(m2)
    print(f"\n權重形狀：{[w.shape for w in W1]}")
    assert [w.shape for w in W1] == [w.shape for w in W2], "兩個模型架構不一致"

    print("\n===== Parent baseline =====")
    r1 = evaluate(m1, args.n_eval, args.difficulty, "agent1")
    r2 = evaluate(m2, args.n_eval, args.difficulty, "agent2")

    obs_batch = pt.as_tensor(
        np.stack([env.observation_space.sample() for _ in range(64)]), dtype=pt.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    if args.method in ("paper", "both"):
        for col_dir in ("paper", "fixed"):
            label_cd = "論文欄位方向" if col_dir == "paper" else "修正欄位方向"
            print(f"\n===== 對齊（{label_cd}）=====")
            aligned, perms = align_nodes([copy.deepcopy(W1), copy.deepcopy(W2)],
                                         stop_layer_k=3, permute_last=True, col_dir=col_dir)

            d = check_permutation_consistency(m2, perms, obs_batch, col_dir=col_dir)
            print(f"  排列自洽檢驗：對 agent2 單獨套用這組排列，動作輸出最大變化 = {d:.3e}")
            print(f"    {'✓ 列與欄方向一致（輸出不變）' if d < 1e-5 else '✗ 輸出變了，深層拿到的是被打亂的輸入'}")

            fused_W = fuse(aligned)

            # bias 照同一組排列對齊後平均（論文沒做，這裡當對照）
            aB2 = [b.copy() for b in B2]
            for i in range(2):
                aB2[i] = aB2[i][np.argsort(perms[i])]
            fused_B = [(B1[i] + aB2[i]) / 2.0 for i in range(3)]

            for bias_mode, fb in (("bias_random", None), ("bias_fused", fused_B)):
                tag = f"{col_dir}|{bias_mode}"
                fm = build_fused(env, fused_W, fused_B=fb)
                note = "照論文，bias 維持新模型隨機值" if fb is None else "bias 也對齊後平均"
                print(f"  --- {tag}（{note}）---")
                results[tag] = evaluate(fm, args.n_eval, args.difficulty, tag)
                path = os.path.join(args.out_dir, f"ot_paper_fused_{col_dir}_{bias_mode}.pkl")
                with open(path, "wb") as f:
                    cloudpickle.dump(fm.policy, f)
                print(f"      已存：{path}")

    if args.method in ("ot_iter", "both"):
        print(f"\n===== ot_iter（迭代 OT 雙通道拼接，{args.ot_iters} 輪，ot_frac={args.ot_frac}）=====")
        spliced_W, spliced_B = iterative_ot_splice(
            W1, W2, B1, B2, n_iters=args.ot_iters, ot_frac=args.ot_frac)

        for bias_mode, fb in (("bias_random", None), ("bias_spliced", spliced_B)):
            tag = f"ot_iter|{bias_mode}"
            fm = build_fused(env, spliced_W, fused_B=fb)
            note = "跟論文一樣，bias 維持新模型隨機值" if fb is None else "bias 也用對半拆拼接"
            print(f"  --- {tag}（{note}）---")
            results[tag] = evaluate(fm, args.n_eval, args.difficulty, tag)
            path = os.path.join(args.out_dir, f"ot_iter_fused_{bias_mode}.pkl")
            with open(path, "wb") as f:
                cloudpickle.dump(fm.policy, f)
            print(f"      已存：{path}")

    print("\n" + "=" * 62)
    print(f"{'版本':<34}{'mean':>10}{'std':>10}")
    print("-" * 62)
    print(f"{'agent1 (' + os.path.basename(args.agent1) + ')':<34}{r1[0]:>10.2f}{r1[1]:>10.2f}")
    print(f"{'agent2 (' + os.path.basename(args.agent2) + ')':<34}{r2[0]:>10.2f}{r2[1]:>10.2f}")
    for k, (m, s) in results.items():
        print(f"{k:<34}{m:>10.2f}{s:>10.2f}")
    print("=" * 62)
    env.close()


if __name__ == "__main__":
    main()
