# -*- coding: utf-8 -*-
"""
用 Optuna 幫 ot_progressive_iter 找參數。

前面的「OT → 蒸餾」是離線做的，可以拿「PPO 前」的模型當便宜的替代指標，
不用每組參數都接一次真實環境 PPO。所以這裡拆成兩階段：

  階段 1（搜尋）：每個 trial 只跑 OT + 蒸餾（final_finetune_steps=0），直接評估
    「PPO 前」的模型，用固定 seed、多個難度的平均 reward 當分數。
    蒸餾是整條流程裡最慢的部分（完整資料一輪約 9 分鐘，預設 4 層×10 輪≈6 小時），
    所以預設只抽 5% 蒸餾資料（--distill_frac），一個 trial 才壓得到十幾二十分鐘。
  階段 2（驗證）：挑分數前 top_k 名，載入階段 1 存下的模型，接短版 PPO 微調
    （預設 10 萬步、起始難度 0.5，也就是 --evolved_all 那套實測較好的設定）再評估。
    --start_difficulty 可以給多個值，每個候選模型會各跑一次，順便比較起始難度。

  驗證（階段 1 之後）：前 --val_top_k 名改用一批「搜尋時從沒用過」的地形（--val_seed）
    各難度再評 --val_episodes 局，用這個分數排名，避開「在搜尋地形上挑最高分」的挑選偏差。
    --val_train_seeds N（N>0）時，不沿用搜尋時訓練出的那一個模型，而是每組參數用 N 個
    新的訓練種子（1000, 1001, ...）各重新訓練一次、各自在保留地形上評估，取平均排名——
    同時排除「地形的運氣」和「訓練的運氣」。N=0 為舊行為（只換地形重評同一個模型）。
    階段 2 的候選也改用驗證分數排序。

--space 選搜尋範圍與評估設定：
  v1＝第一次調參的設定（範圍較寬、4 個難度×5 局、所有難度共用同一組 seed），保留用來重現舊 study。
  v2（預設）＝根據 v1 結果縮小範圍、alpha_init 含 0（＝不做 OT）、7 個難度×25 局、每個難度各自一組 seed。

study 存在 sqlite，中斷後用同一個 --study_name 重跑會接著跑，不會從頭來。

用法：
  python tune_ot_progressive_iter.py --n_trials 40 --skip_finetune                       # v2 搜尋 + 驗證
  python tune_ot_progressive_iter.py --n_trials 40 --skip_finetune --distill_scope layer --chain_T
  python tune_ot_progressive_iter.py --n_trials 0 --top_k 3 --start_difficulty 0.3 0.5 0.7   # 只跑階段 2
  python tune_ot_progressive_iter.py --skip_finetune                                          # 只跑階段 1
"""
import os, sys, csv, json, math, random, argparse, importlib.util
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import gym
import cloudpickle
import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env.custom_env  # noqa: F401
from stable_baselines3 import PPO

ENV_ID = "BipedalWalkerCustom-v0"


def _load_bw_main():
    """主程式住在 BipedalWalker-v3.py，檔名有橫線不能直接 import。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BipedalWalker-v3.py")
    spec = importlib.util.spec_from_file_location("bw_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bw_main"] = mod
    spec.loader.exec_module(mod)
    return mod


bw = _load_bw_main()


def load_any(path, env):
    """跟 --test_ties 的 _load_any 相同：.pkl 是 cloudpickle 存的 policy，其餘當 SB3 zip。"""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = cloudpickle.load(f)
        policy = obj.policy if isinstance(obj, PPO) else obj
        model = PPO("MlpPolicy", env, verbose=0)
        model.policy = policy.to(model.device)
        return model
    load_path = path[:-4] if path.endswith(".zip") else path
    return PPO.load(load_path, env=env, device="cpu")


def make_seeds(base_seed, difficulties, n, per_difficulty=True):
    """
    per_difficulty=True：每個難度各自一組 seed（算法同 eval_per_difficulty.fixed_seeds），
    不同難度不會重複用同幾張地形。False：所有難度共用同一組（v1 的舊行為，重現舊 study 用）。
    """
    if not per_difficulty:
        rng = random.Random(base_seed)
        shared = [rng.randint(0, 2**31 - 1) for _ in range(n)]
        return {d: shared for d in difficulties}
    out = {}
    for d in difficulties:
        rng = random.Random(base_seed * 1000 + int(round(d * 10)))
        out[d] = [rng.randint(0, 2**31 - 1) for _ in range(n)]
    return out


def evaluate(model, difficulties, seeds):
    """seeds：{難度: [seed,...]}。每個 trial 都用同一組，才是在同樣的地形上比較。"""
    per_diff = {}
    for d in difficulties:
        e = gym.make(ENV_ID, difficulty=d)
        rewards = []
        for seed in seeds[d]:
            obs = e.reset(seed=seed)
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
        per_diff[d] = float(np.mean(rewards))
    return float(np.mean(list(per_diff.values()))), per_diff


def build_child(dad, mom, env):
    """跟 --test_ties 建 child 的方式相同：雙通道拼接，交叉通道留給 progressive_iterative_ot_evolve 寫。"""
    child_policy = bw.create_dual_channel_policy(dad.policy, mom.policy)
    child_model = PPO("MlpPolicy", env, verbose=0)
    child_model.policy = child_policy.to(child_model.device)
    child_model.policy.optimizer = torch.optim.Adam(child_model.policy.parameters(), lr=1e-4)
    return child_model


def seed_everything(seed):
    """
    固定訓練的隨機性：蒸餾的 DataLoader(shuffle=True) 用的是 PyTorch 全域亂數，
    child 建立時 PPO 的隨機初始化也是，所以每個 trial 開始前都重設一次。
    搭配 main() 裡的 torch.use_deterministic_algorithms，同一個 trial 重跑會得到相同的權重。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gamma_from_end(alpha_init, alpha_end, n_rounds):
    """直接搜「最後一輪的 alpha」，換算回 gamma，改 n_rounds 時衰減曲線才不會跑掉。"""
    if n_rounds <= 1 or alpha_init <= 0:
        return 1.0  # alpha_init=0 代表完全不做 OT，gamma 沒有意義
    return (alpha_end / alpha_init) ** (1.0 / (n_rounds - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dad", default="./best_model/model2.zip")
    ap.add_argument("--mom", default="./best_model/model3.zip")
    ap.add_argument("--distill_pt", default="./logs/ga_eval/mtkd_continuous.pt(2)")
    ap.add_argument("--distill_frac", type=float, default=0.05,
                    help="每輪蒸餾只用多少比例的資料（完整 13GB 一輪約 9 分鐘；0.05 約 30 秒）。"
                         "最後印出的指令會帶同樣的 --progressive_distill_frac，抽到的是同一批樣本")
    ap.add_argument("--n_trials", type=int, default=40, help="這次要新跑幾個 trial（0＝只跑階段 2）")
    ap.add_argument("--space", default="v2", choices=["v1", "v2"],
                    help="搜尋範圍與評估設定（見檔頭說明）。v1 重現第一次調參；v2 為預設")
    ap.add_argument("--study_name", default=None, help="預設 v1=ot_prog_iter、v2=ot_prog_iter_v2")
    ap.add_argument("--out_dir", default=None,
                    help="預設 v1=./logs/optuna_ot_prog_iter、v2=./logs/optuna_ot_prog_iter_v2")
    ap.add_argument("--eval_difficulties", type=float, nargs="+", default=None,
                    help="階段 1/2 評估用的難度，分數取平均。預設 v1=0/0.3/0.6/0.9、v2=0/0.3/0.6/0.7/0.8/0.9/1.0")
    ap.add_argument("--eval_episodes", type=int, default=None, help="每個難度評幾局（固定 seed）。預設 v1=5、v2=25")
    ap.add_argument("--eval_seed", type=int, default=1234, help="搜尋用評估地形的 seed 基底")
    ap.add_argument("--seed", type=int, default=0, help="TPE sampler 的種子")
    ap.add_argument("--distill_scope", default="all", choices=["all", "layer"],
                    help="傳給 progressive_iterative_ot_evolve：all=每輪蒸餾全部層；layer=只蒸餾剛 OT 過的這一層")
    ap.add_argument("--train_seed", type=int, default=0,
                    help="每個 trial 開始前用這個種子固定訓練的隨機性（蒸餾資料打亂順序、網路初始化），"
                         "所有 trial 用同一個種子，結果可以重現、參數之間的比較也比較公平。給 -1 = 不固定（舊行為）")
    ap.add_argument("--fixed_ot_frac", type=float, default=None,
                    help="給了就把 ot_frac 固定成這個值、不參與搜尋（例如 1.0＝整塊交叉通道都寫 OT 值）。"
                         "預設 study 名稱／資料夾會自動加上 _otfrac<值> 後綴，不會跟搜尋 ot_frac 的 study 混在一起")
    ap.add_argument("--chain_T", action="store_true", help="傳給 progressive_iterative_ot_evolve：上一層的 T 影響下一層")
    # 驗證
    ap.add_argument("--val_top_k", type=int, default=5, help="搜尋完取前幾名做保留地形驗證（0＝不驗證）")
    ap.add_argument("--val_episodes", type=int, default=100, help="驗證時每個難度評幾局")
    ap.add_argument("--val_train_seeds", type=int, default=0,
                    help="驗證時每組參數用幾個訓練種子重新訓練（種子 1000、1001、...，跟搜尋用的 --train_seed 分開）。"
                         "0＝舊行為：直接拿搜尋時訓練出的模型換地形重評")
    ap.add_argument("--val_seed", type=int, default=777,
                    help="驗證地形的 seed 基底，要跟 --eval_seed 不同，也不要用最終報告的 2026")
    # 階段 2
    ap.add_argument("--skip_finetune", action="store_true", help="只跑階段 1")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--finetune_steps", type=int, default=100_000)
    ap.add_argument("--start_difficulty", type=float, nargs="+", default=[0.5],
                    help="PPO 微調的起始訓練難度，可給多個值互相比較（預設 0.5，跟 --evolved_all 相同）")
    ap.add_argument("--diff_lr_scale", type=float, default=None)
    ap.add_argument("--diff_lr_target", default="cross", choices=["cross", "pure"])
    args = ap.parse_args()
    v1 = args.space == "v1"
    frac_tag = "" if args.fixed_ot_frac is None else f"_otfrac{args.fixed_ot_frac:g}"
    if args.study_name is None:
        args.study_name = ("ot_prog_iter" if v1 else "ot_prog_iter_v2") + frac_tag
    if args.out_dir is None:
        args.out_dir = ("./logs/optuna_ot_prog_iter" if v1 else "./logs/optuna_ot_prog_iter_v2") + frac_tag
    if args.eval_difficulties is None:
        args.eval_difficulties = [0.0, 0.3, 0.6, 0.9] if v1 else [0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0]
    if args.eval_episodes is None:
        args.eval_episodes = 5 if v1 else 25
    assert args.val_seed != args.eval_seed, "驗證地形必須跟搜尋地形不同"
    print(f"space={args.space}  study={args.study_name}  out_dir={args.out_dir}  "
          f"評估 {args.eval_difficulties} × {args.eval_episodes} 局  "
          f"distill_scope={args.distill_scope}  chain_T={args.chain_T}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.train_seed >= 0:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"固定訓練種子 train_seed={args.train_seed}（每個 trial 開始前重設，開啟確定性模式）")
    eval_seeds = make_seeds(args.eval_seed, args.eval_difficulties, args.eval_episodes, per_difficulty=not v1)

    env = gym.make(ENV_ID, difficulty=0.0)
    print(f"載入 dad：{args.dad}\n載入 mom：{args.mom}")
    dad = load_any(args.dad, env)
    mom = load_any(args.mom, env)

    # 13GB 蒸餾資料只讀一次、抽樣一次，所有 trial 共用；抽樣用固定 seed，跟主程式
    # --progressive_distill_frac 抽到的是同一批
    distill_blob = None
    if args.n_trials > 0 or (args.val_top_k > 0 and args.val_train_seeds > 0):
        print(f"讀取蒸餾資料：{args.distill_pt}")
        full = torch.load(args.distill_pt, map_location="cpu")
        distill_blob = bw._subsample_distill_blob(full, args.distill_frac)
        del full

    storage = f"sqlite:///{os.path.join(args.out_dir, 'study.db')}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # 先排幾個固定的 trial 當基準線
    if len(study.trials) == 0:
        if v1:
            study.enqueue_trial({"alpha_init": 1.0, "alpha_end": 0.05, "n_rounds": 10,
                                 "distill_epochs": 5, "distill_lr": 0.008, "ot_frac": 1.0})
        else:
            # v1 的 trial 35，以及同一組參數關掉 OT（alpha_init=0）——直接在搜尋裡放一組 OT 對照
            t35 = {"alpha_init": 0.8315, "alpha_end": 0.1445, "n_rounds": 3,
                   "distill_epochs": 7, "distill_lr": 0.0088, "ot_frac": 0.658}
            if args.fixed_ot_frac is not None:
                t35.pop("ot_frac")  # ot_frac 不搜尋時，不能放進 enqueue 的參數裡
            study.enqueue_trial(t35)
            study.enqueue_trial({**t35, "alpha_init": 0.0})

    # ── 階段 1：只跑 OT + 蒸餾，評估 PPO 前的模型 ─────────────────────
    def objective(trial):
        if v1:
            alpha_init = trial.suggest_float("alpha_init", 0.3, 1.0)
            alpha_end = trial.suggest_float("alpha_end", 0.01, 0.3, log=True)
            n_rounds = trial.suggest_int("n_rounds", 3, 10)
            distill_epochs = trial.suggest_int("distill_epochs", 2, 10)
            distill_lr = trial.suggest_float("distill_lr", 1e-3, 1e-2, log=True)
            ot_frac = trial.suggest_float("ot_frac", 0.5, 1.0) if args.fixed_ot_frac is None else args.fixed_ot_frac
        else:
            # 根據 v1：n_rounds / distill_epochs 重要度都只有 0.03，縮小並偏向便宜的一端；
            # distill_lr 最重要、前幾名集中 0.0045~0.009，上限放寬；ot_frac 前幾名 0.54~0.67，下限往下探；
            # alpha_init 含 0，讓 Optuna 自己判斷 OT 有沒有用
            alpha_init = trial.suggest_float("alpha_init", 0.0, 1.0)
            alpha_end = trial.suggest_float("alpha_end", 0.01, 0.3, log=True)
            n_rounds = trial.suggest_int("n_rounds", 3, 5)
            distill_epochs = trial.suggest_int("distill_epochs", 2, 7)
            distill_lr = trial.suggest_float("distill_lr", 3e-3, 1.5e-2, log=True)
            ot_frac = trial.suggest_float("ot_frac", 0.4, 0.8) if args.fixed_ot_frac is None else args.fixed_ot_frac
        alpha_end = min(alpha_end, alpha_init)
        gamma = gamma_from_end(alpha_init, alpha_end, n_rounds)
        trial.set_user_attr("alpha_gamma", gamma)
        trial.set_user_attr("ot_frac", ot_frac)
        trial.set_user_attr("train_seed", args.train_seed)
        if args.train_seed >= 0:
            seed_everything(args.train_seed)

        child = build_child(dad, mom, env)
        bw.progressive_iterative_ot_evolve(
            child, dad.policy, mom.policy, args.distill_pt, env=env,
            n_rounds=n_rounds, distill_epochs=distill_epochs, distill_lr=distill_lr,
            final_finetune_steps=0, ot_frac=ot_frac, device=str(child.device),
            alpha_init=alpha_init, alpha_gamma=gamma, pre_finetune_save_path=None,
            preloaded_blob=distill_blob,
            distill_scope=args.distill_scope, chain_T=args.chain_T,
        )
        score, per_diff = evaluate(child, args.eval_difficulties, eval_seeds)
        trial.set_user_attr("per_difficulty", {str(k): v for k, v in per_diff.items()})

        path = os.path.join(args.out_dir, f"trial_{trial.number}_preppo.pkl")
        with open(path, "wb") as f:
            cloudpickle.dump(child.policy, f)
        trial.set_user_attr("model_path", path)
        print(f"[trial {trial.number}] score={score:.2f}  {per_diff}")
        return score

    if args.n_trials > 0:
        study.optimize(objective, n_trials=args.n_trials)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("還沒有完成的 trial")
        return
    done.sort(key=lambda t: t.value, reverse=True)

    print("\n===== 階段 1 排名（PPO 前）=====")
    for t in done[:10]:
        print(f"  #{t.number:3d}  score={t.value:7.2f}  {t.params}  gamma={t.user_attrs.get('alpha_gamma'):.4f}")
    if len(done) >= 5:
        try:
            imp = optuna.importance.get_param_importances(study)
            print("\n參數重要度：", {k: round(v, 3) for k, v in imp.items()})
        except Exception as e:
            print(f"(參數重要度算不出來：{e})")

    # ── 驗證：前 val_top_k 名在保留地形上重評，用這個分數選最後的參數 ──────
    if args.val_top_k > 0:
        val_seeds = make_seeds(args.val_seed, args.eval_difficulties, args.val_episodes)
        print(f"\n===== 驗證：前 {args.val_top_k} 名在保留地形（seed 基底 {args.val_seed}，"
              f"{args.eval_difficulties} × {args.val_episodes} 局）上重評 =====")
        # 已完成的 trial 不能再改 user_attr，驗證結果改存在 study 層級：
        # study.user_attrs["val::<seed>::<局數>"] = {trial 編號: {"score":..., "per_difficulty":...}}
        if args.val_train_seeds > 0:
            # 多種子驗證：每組參數用 N 個新的訓練種子各重新訓練一次，取平均
            train_seeds = [1000 + i for i in range(args.val_train_seeds)]
            val_key = f"valms::{args.val_seed}::{args.val_episodes}::{args.val_train_seeds}"
            val_results = dict(study.user_attrs.get(val_key, {}))
            print(f"每組參數用訓練種子 {train_seeds} 各重新訓練一次")
            for t in done[:args.val_top_k]:
                rec = dict(val_results.get(str(t.number), {"seeds": {}}))
                rec["seeds"] = dict(rec.get("seeds", {}))
                p = t.params
                ot_frac = t.user_attrs.get("ot_frac", p.get("ot_frac", args.fixed_ot_frac))
                gamma = t.user_attrs["alpha_gamma"]
                for sd in train_seeds:
                    if str(sd) in rec["seeds"]:
                        continue  # 續跑時，已經完成的種子不重算
                    seed_everything(sd)
                    child = build_child(dad, mom, env)
                    bw.progressive_iterative_ot_evolve(
                        child, dad.policy, mom.policy, args.distill_pt, env=env,
                        n_rounds=p["n_rounds"], distill_epochs=p["distill_epochs"], distill_lr=p["distill_lr"],
                        final_finetune_steps=0, ot_frac=ot_frac, device=str(child.device),
                        alpha_init=p["alpha_init"], alpha_gamma=gamma, pre_finetune_save_path=None,
                        preloaded_blob=distill_blob,
                        distill_scope=args.distill_scope, chain_T=args.chain_T,
                    )
                    vs, vpd = evaluate(child, args.eval_difficulties, val_seeds)
                    mpath = os.path.join(args.out_dir, f"trial_{t.number}_valseed{sd}_preppo.pkl")
                    with open(mpath, "wb") as f:
                        cloudpickle.dump(child.policy, f)
                    rec["seeds"][str(sd)] = {"score": vs, "per_difficulty": {str(k): v for k, v in vpd.items()},
                                             "model_path": mpath}
                    sc = [v["score"] for v in rec["seeds"].values()]
                    rec["score"], rec["std"] = float(np.mean(sc)), float(np.std(sc))
                    val_results[str(t.number)] = rec
                    study.set_user_attr(val_key, val_results)
                    print(f"  #{t.number:3d}  訓練種子 {sd}：驗證 {vs:7.2f}  {vpd}", flush=True)
                sc = [rec["seeds"][str(sd)]["score"] for sd in train_seeds]
                print(f"  #{t.number:3d}  搜尋 {t.value:7.2f} → 驗證平均 {np.mean(sc):7.2f} ± {np.std(sc):.2f}  "
                      f"（{' / '.join(f'{x:.1f}' for x in sc)}）", flush=True)
        else:
            val_key = f"val::{args.val_seed}::{args.val_episodes}"
            val_results = dict(study.user_attrs.get(val_key, {}))
            for t in done[:args.val_top_k]:
                if str(t.number) in val_results:
                    continue  # 之前驗證過，續跑時不重算
                with open(t.user_attrs["model_path"], "rb") as f:
                    policy = cloudpickle.load(f)
                vm = PPO("MlpPolicy", env, verbose=0)
                vm.policy = policy.to(vm.device)
                vs, vpd = evaluate(vm, args.eval_difficulties, val_seeds)
                val_results[str(t.number)] = {"score": vs, "per_difficulty": {str(k): v for k, v in vpd.items()}}
                study.set_user_attr(val_key, val_results)
                print(f"  #{t.number:3d}  搜尋 {t.value:7.2f} → 驗證 {vs:7.2f}  {vpd}")
        by_num = {t.number: t for t in done}
        validated = sorted([by_num[int(n)] for n in val_results if int(n) in by_num],
                           key=lambda t: val_results[str(t.number)]["score"], reverse=True)
        print("\n===== 驗證排名（用這個選最後的參數）=====")
        for t in validated:
            r = val_results[str(t.number)]
            spread = f" ± {r['std']:.2f}（{len(r['seeds'])} 個訓練種子）" if "seeds" in r else ""
            print(f"  #{t.number:3d}  驗證 {r['score']:7.2f}{spread}  搜尋 {t.value:7.2f}  {t.params}")
        rest = [t for t in done if str(t.number) not in val_results]
        done = validated + rest  # 階段 2 的候選改用驗證分數排序

    if args.skip_finetune:
        return

    # ── 階段 2：前 top_k 名接短版 PPO，逐一比較起始難度 ──────────────
    # AutoDifficultyCallback 會把 hard seed 寫回檔案；調參跑很多次，用副本避免汙染主池
    hs_copy = os.path.join(args.out_dir, "hard_seeds_tune.json")
    if not os.path.exists(hs_copy):
        src = "./logs/hard_seeds.json"
        hs = json.load(open(src, encoding="utf-8")) if os.path.exists(src) else {}
        json.dump(hs, open(hs_copy, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 有開差異化學習率時，模型檔名和結果 CSV 都帶上後綴，不會蓋掉沒開的那一輪
    lr_tag = "" if args.diff_lr_scale is None else f"_difflr{args.diff_lr_target}{args.diff_lr_scale}"
    rows = []
    for t in done[:args.top_k]:
        for sd in args.start_difficulty:
            print(f"\n===== 階段 2：trial #{t.number}，起始難度 {sd}，PPO {args.finetune_steps} 步 =====")
            train_env = gym.make(ENV_ID, difficulty=sd)
            with open(t.user_attrs["model_path"], "rb") as f:
                policy = cloudpickle.load(f)
            model = PPO("MlpPolicy", train_env, verbose=0,
                        seed=args.train_seed if args.train_seed >= 0 else None)
            model.policy = policy.to(model.device)
            handles = bw._setup_finetune_optimizer(model, args.diff_lr_scale, args.diff_lr_target)
            cb = bw.AutoDifficultyCallback(
                train_env, None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,
                shared_flags=None, cooldown_steps=0, hardseed_save_path=hs_copy,
            )
            model.learn(total_timesteps=args.finetune_steps, callback=[cb], progress_bar=True)
            for h in handles:
                h.remove()
            train_env.close()

            score, per_diff = evaluate(model, args.eval_difficulties, eval_seeds)
            out = os.path.join(args.out_dir, f"trial_{t.number}_sd{sd}_{args.finetune_steps}{lr_tag}.pkl")
            with open(out, "wb") as f:
                cloudpickle.dump(model.policy, f)
            print(f"  PPO 前 {t.value:.2f} → PPO 後 {score:.2f}  {per_diff}")
            rows.append({"trial": t.number, "start_difficulty": sd, "preppo_score": round(t.value, 2),
                         "finetuned_score": round(score, 2), "model_path": out,
                         **t.params, "alpha_gamma": t.user_attrs.get("alpha_gamma")})

    csv_path = os.path.join(args.out_dir, f"finetune_results{lr_tag}.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)

    rows.sort(key=lambda r: r["finetuned_score"], reverse=True)
    print("\n===== 階段 2 排名（PPO 後）=====")
    for r in rows:
        print(f"  trial #{r['trial']:3d}  起始難度 {r['start_difficulty']}  "
              f"PPO前 {r['preppo_score']:7.2f} → PPO後 {r['finetuned_score']:7.2f}")
    b = rows[0]
    print("\n最佳設定換成 --test_ties 指令：")
    print(f"  python BipedalWalker-v3.py --test_ties --crossover ot_progressive_iter "
          f"--dad {args.dad} --mom {args.mom} "
          f"--start_difficulty {b['start_difficulty']} "
          f"--progressive_n_rounds {b['n_rounds']} "
          f"--progressive_ot_alpha_init {b['alpha_init']:.4f} "
          f"--progressive_ot_gamma {b['alpha_gamma']:.4f} "
          f"--progressive_distill_epochs {b['distill_epochs']} "
          f"--progressive_distill_lr {b['distill_lr']:.5f} "
          f"--progressive_distill_frac {args.distill_frac} "
          f"--ot_frac {b.get('ot_frac', args.fixed_ot_frac):.3f} "
          f"--progressive_distill_scope {args.distill_scope} "
          + ("--progressive_chain_T " if args.chain_T else "")
          + f"--progressive_steps_per_round {args.finetune_steps}"
          + ("" if args.diff_lr_scale is None
             else f" --diff_lr_scale {args.diff_lr_scale} --diff_lr_target {args.diff_lr_target}"))
    print(f"\n結果已寫入 {csv_path}")


if __name__ == "__main__":
    main()
