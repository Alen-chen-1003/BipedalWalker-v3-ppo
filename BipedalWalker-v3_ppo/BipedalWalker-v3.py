import env.custom_env 
import hashlib
import shutil
import traceback  # 匯入 traceback 模組以便印出詳細錯誤
import seaborn as sns
import csv
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from typing import Union
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Set, Optional
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
import cloudpickle, os
from typing import List, Tuple, Dict, Any
import math, torch
import copy, torch
from typing import Tuple
import torch.nn as nn
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict
import torch
import os
import gym
import numpy as np
import time
import json
import random
   # 确保 custom_env.py 中的 register() 被执行
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback,
    BaseCallback
)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # 用於顯示進度條，更友善
import pandas as pd
from stable_baselines3.common.logger import configure 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 修改：make_env 加一个 difficulty 参数
class CurriculumCallback(BaseCallback):
    """
    定期將 DummyVecEnv 的難度，優先複習表現較差的難度，並自動複習 hardseed。
    複習結束後自動切回最高難度，並自動測試清理 hardseed。
    支援 hardseed reward 動態更新 + 機率式複習 seed
    """
    def __init__(self, difficulty_list, eval_env, hardseeds_path: str = "./logs/hard_seeds.json",
                 switch_freq=50000, review_steps=10000,
                 hardseed_reward_threshold=100, hardseed_n_eval=2,
                 verbose=1, shared_flags=None, cooldown_steps=30000,seeds_per_group=5):
        super().__init__(verbose)
        self.shared_flags = shared_flags or {}
        self.difficulty_list = difficulty_list
        self.eval_env = eval_env
        # hard_seeds 應為 {難度: {seed: reward, ...}, ...}
        self.hardseed_path = hardseeds_path
        with open(self.hardseed_path, "r") as f:
            # 檔案格式：{ "0.50": {"seed1": 123.4, "seed2": 56.7}, ... }
            self.hard_seeds = json.load(f)
        self.switch_freq = switch_freq
        self.review_steps = review_steps
        self.last_switch = 0
        self.reviewing = False
        self.review_countdown = 0
        self.bad_hardseeds = []
        self.current_seed_idx = 0
        self.review_step_count = 0
        self.prev_difficulty = None
        self.hardseed_reward_threshold = hardseed_reward_threshold
        self.hardseed_n_eval = hardseed_n_eval
        self.cooldown_steps = cooldown_steps
        self.next_allowed_switch = 0  # 下一次允許觸發複習的步數
        self.seeds_per_group = seeds_per_group
    def _sample_seeds_across_all_diffs(self) -> list[int]:
        bucket1, bucket2, bucket3, bucket4 = [], [], [], []   # 200-250, 125-200, 0-120, <0
        for diff_dict in self.hard_seeds.values():
            for seed_str, reward in diff_dict.items():
                r = float(reward)
                if 200 <= r < 250:
                    bucket1.append(int(seed_str))
                elif 125 <= r < 200:
                    bucket2.append(int(seed_str))
                elif 0   <= r < 120:
                    bucket3.append(int(seed_str))
                elif r < 0:
                    bucket4.append(int(seed_str))

        sampled = []
        for bucket in (bucket1, bucket2, bucket3, bucket4):
            if bucket:
                sampled += random.sample(bucket, min(self.seeds_per_group, len(bucket)))
        return sampled

    def _on_step(self) -> bool:
        # ───────────────────────── 冷卻中 ─────────────────────────
        if self.n_calls < self.next_allowed_switch:
            return True

        # ───────────────   正在 reviewing 模式   ────────────────
        if self.reviewing:
            self.shared_flags["reviewing"] = True
            steps_per_seed = max(1, self.review_steps // len(self.bad_hardseeds))
            seed = self.bad_hardseeds[self.current_seed_idx]
            self.training_env.env_method("set_seed", seed)
            self.review_step_count += 1

            if self.review_step_count >= steps_per_seed:
                self.current_seed_idx += 1
                self.review_step_count = 0

                if self.current_seed_idx >= len(self.bad_hardseeds):
                    self._test_and_remove_badseeds()
                    print(f"[HardSeedReview] 複習結束，切回難度 {self.prev_difficulty:.2f}")
                    self.training_env.env_method("set_difficulty", self.prev_difficulty)
                    self.training_env.env_method("set_seed", None)
                    self.reviewing = False
                    self.shared_flags["reviewing"] = False
                    self.last_switch = self.n_calls
            return True

        # ──────────────────   是否到時間啟動複習？   ────────────────
        if (self.n_calls - self.last_switch) <= self.switch_freq:
            return True

        # ============  (A) 準備進入 hard-seed 複習  =============
        self.prev_difficulty = self.training_env.env_method("get_difficulty")[0]
        sampled_seeds = self._sample_seeds_across_all_diffs()     # ←★ 新抽樣方式
        if not sampled_seeds:
            print("[HardSeedReview] 找不到任何待複習的 hard-seed，跳過。")
            self.last_switch = self.n_calls
            self.next_allowed_switch = self.n_calls + self.cooldown_steps
            with open(self.hardseed_path, "w") as f:          # ★
                json.dump(self.hard_seeds, f, ensure_ascii=False, indent=2)
            return True

        # ============  (B) 對抽到的 seed 先離線評估一次  ==========
        bad_seeds = []
        for seed in sampled_seeds:
            avg_reward = self._evaluate_seed(seed)

            # 把最新 reward 寫回檔案結構
            for diff in self.hard_seeds:
                if str(seed) in self.hard_seeds[diff]:
                    self.hard_seeds[diff][str(seed)] = avg_reward

            if avg_reward >= 250:
                for diff in self.hard_seeds:
                    self.hard_seeds[diff].pop(str(seed), None)
                print(f"[HardSeedReview] seed {seed} 已過關，自動移除。")
            elif avg_reward < self.hardseed_reward_threshold:
                bad_seeds.append(seed)
                print(f"[HardSeedReview] seed {seed} 沒過關，加入 badseed。")

        # ============  (C) 真正進入 reviewing  ============
        if bad_seeds:
            self.shared_flags["reviewing"] = True  
            print(f"\n[HardSeedReview] 進入複習，共 {len(bad_seeds)} 個 seed\n")
            self.bad_hardseeds = bad_seeds
            self.current_seed_idx = 0
            self.review_step_count = 0
            self.reviewing = True
            self.training_env.env_method("set_seed", bad_seeds[0])
        else:
            print("[HardSeedReview] 沒有需要複習的 hard-seed，本次跳過。")

        self.last_switch        = self.n_calls
        self.next_allowed_switch = self.n_calls + self.cooldown_steps
        return True

    def _evaluate_seed(self, seed):
        self.eval_env.unwrapped.set_seed(seed)
        total_rewards = []
        for _ in range(self.hardseed_n_eval):
            obs, _ = self.eval_env.reset()
            done = False
            total = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total += reward
            total_rewards.append(total)
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"[HardSeedTest] seed {seed}, 平均回報 {avg_reward:.2f}")
        return avg_reward

    def _test_and_remove_badseeds(self):
        """複習後再測試 bad_hardseeds，會的自動移除"""
        to_remove = []
        for seed in self.bad_hardseeds:
            avg_reward = self._evaluate_seed(seed)
            # 寫回最新 reward
            for d in self.hard_seeds:
                if str(seed) in self.hard_seeds[d]:
                    self.hard_seeds[d][str(seed)] = avg_reward
            if avg_reward >= self.hardseed_reward_threshold:
                for d in self.hard_seeds:
                    if str(seed) in self.hard_seeds[d]:
                        self.hard_seeds[d].pop(str(seed))
                        print(f"[HardSeed] seed {seed} 學會了，已移除。")
                to_remove.append(seed)
        self.bad_hardseeds = [s for s in self.bad_hardseeds if s not in to_remove]
        with open(self.hardseed_path, "w") as f:
            json.dump(self.hard_seeds, f, ensure_ascii=False, indent=2)
class AutoDifficultyCallback(BaseCallback):
    def __init__(self, eval_env, hard_seeds, eval_freq=10000, reward_threshold=250, increase=0.05, verbose=1,
                 difficulty_list=None, shared_flags=None, cooldown_steps=30000, hardseed_save_path="./logs/hard_seeds.json"):
        super().__init__(verbose)
        self.shared_flags = shared_flags or {}
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.reward_threshold = reward_threshold
        self.increase = increase
        self.current_difficulty = eval_env.unwrapped.difficulty
        # {難度: {seed: reward}}
        if hard_seeds is None:
            with open(hardseed_save_path, "r") as f:
                self.hard_seeds = json.load(f)
        else:
            self.hard_seeds = hard_seeds
        self.hardseed_save_path = hardseed_save_path

        self.difficulty_list = list(self.hard_seeds.keys())
        self.cooldown_steps = cooldown_steps
        self.next_allowed_upgrade = 0
        

    def save_hard_seeds(self):
        # 存成 {難度: {seed: reward}}
        import json
        with open(self.hardseed_save_path, "w") as f:
            json.dump(self.hard_seeds, f, ensure_ascii=False, indent=2)
        print(f"[AutoDiff] 已保存 hard_seeds 至 {self.hardseed_save_path}")

    def _on_step(self) -> bool:
        if self.n_calls < self.next_allowed_upgrade or self.shared_flags.get("reviewing", False):
            return True
        # 每 eval_freq 步做一次評估
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            seeds = []
            for _ in range(100):
                obs, info  = self.eval_env.reset()
                seed = info.get("seed", None)
                seeds.append(seed)
                done = False
                total = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total += reward
                rewards.append(total)
            avg_reward = sum(rewards) / len(rewards)
            print(f"[AutoDifficulty] 當前難度 {self.current_difficulty:.2f}，平均回報 {avg_reward:.2f}")
            self.logger.record("curriculum/difficulty", self.current_difficulty)

            if avg_reward > 150:
                print(f"[AutoDiff] 收集 hard seed ）")
                key = str(round(self.current_difficulty, 2))  # 用字串當 key 方便 json 存
                if key not in self.hard_seeds:
                    self.hard_seeds[key] = {}
                for seed, total in zip(seeds, rewards): 
                    if seed is None:
                        print(f"[AutoDiff] 沒有hard seed （回報 {total:.1f}）")
                        continue
                    # 只收錄低於平均值 * 0.5 的 seed，並存 reward
                    if total < avg_reward * 0.5:
                        # 若已經存過該 seed，保留最差 reward（可根據需要改成保存平均）
                        prev = self.hard_seeds[key].get(str(seed), None)
                        if prev is None or total < prev:
                            print(f"[AutoDiff] 收集 hard seed {seed}（回報 {total:.1f}）")
                            self.hard_seeds[key][str(seed)] = total
                self.save_hard_seeds()
            # 超過門檻自動升級
            if avg_reward > self.reward_threshold and self.current_difficulty < 1.0:
                self.current_difficulty = min(self.current_difficulty + self.increase, 1.0)
                self.training_env.env_method('set_difficulty', self.current_difficulty)
                self.eval_env.unwrapped.difficulty = self.current_difficulty
                print(f"[AutoDifficulty] 難度升級為 {self.current_difficulty:.2f}")
                self.next_allowed_upgrade = self.n_calls + self.cooldown_steps  # 設冷卻
        return True
class CustomStopCallback(BaseCallback):
    def __init__(self, eval_env, reward_threshold=300, target_difficulty=0.7, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.reward_threshold = reward_threshold
        self.target_difficulty = target_difficulty
        self.stop_training = False

    def _on_step(self) -> bool:
        # 只有難度達標才評估 reward
        current_difficulty = self.eval_env.unwrapped.difficulty
        if current_difficulty >= self.target_difficulty:
            rewards = []
            for _ in range(10):
                obs, _ = self.eval_env.reset()
                done = False
                total = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total += reward
                rewards.append(total)
            avg_reward = sum(rewards) / len(rewards)
            print(f"[CustomStop] 難度 {current_difficulty:.2f}, 平均回報 {avg_reward:.2f}")
            if avg_reward > self.reward_threshold:
                print("[CustomStop] 達到最高難度且平均回報達標，訓練結束！")
                return False 
        return True
from functools import partial

def make_one_env(env_id: str, rank: int, difficulty: float, seed: int):
    env = gym.make(env_id, difficulty=difficulty)
    env.reset(seed=seed + rank)
    return env

def make_vec_env(env_id, num_envs, difficulty, base_seed, use_subproc=True):
    thunks = [partial(make_one_env, env_id, i, difficulty, base_seed) for i in range(num_envs)]
    return SubprocVecEnv(thunks) if use_subproc else DummyVecEnv(thunks)

def train(config: Dict[str, Any]):
    if torch.cuda.is_available():
        # 建議三個進程就設 0.3 左右（30% * 3 ≈ 90%），避免互相擠爆
        torch.cuda.set_per_process_memory_fraction(0.25, device=0)
    # 修改：改用 custom 环境 ID
    run_id            = config["run_id"]
    env_id = config.get("env_id","BipedalWalkerCustom-v0")
    num_cpu = int(config.get("num_envs",8))
    total_timesteps = int(config.get("total_timestepts",100_000_000_0))
    log_dir = os.path.join("./logs", f"run_{run_id}")
    best_model_dir = os.path.join("./best_model", f"run_{run_id}")
    difficulty_list = [round(0.05 * i, 2) for i in range(1, 21)]  # [0.1,0.2,...,1.0]
    # 你想要的难度（0.0 全平地，1.0 原生 Hardcore） 
    difficulty = 0.0
    base_seed         = int(config.get("base_seed", 42))
    device            = config.get("device", "cuda")  # 同一張卡同時跑多個進程會搶資源，視顯存調整
    use_subproc       = bool(config.get("use_subproc", True))
    # 建立多环境並加上 Monitor
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    vec_env=make_vec_env(env_id,num_envs=num_cpu,difficulty=difficulty,base_seed=base_seed,use_subproc=use_subproc)
    vec_env = VecMonitor(vec_env, log_dir)

    # 設定 SB3 logger
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    ckpt_to_resume = config.get("resume_path", None)
    if ckpt_to_resume and os.path.exists(ckpt_to_resume):
        print(f"[run {run_id}] 🔄 載入舊模型繼續訓練：{ckpt_to_resume}")
        model = PPO.load(ckpt_to_resume, env=vec_env, device=device)
        model.set_env(vec_env)
    else:
        print(f"run {run_id}重新開始新模型")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda",
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
        )
        model.set_logger(new_logger)

    # 評估用環境，也要传 difficulty
    eval_env = gym.make(env_id, difficulty=difficulty)
    difficulty_list = [round(0.05 * i, 2) for i in range(1, 21)]
    hard_seeds = {d: [] for d in difficulty_list}
    shared_flags = {"reviewing": False}
    curriculum_callback = CurriculumCallback(difficulty_list,eval_env, hardseeds_path="./logs/hard_seeds.json",switch_freq=200000, review_steps=60000,hardseed_reward_threshold=250, hardseed_n_eval=10,verbose=1,shared_flags=shared_flags,cooldown_steps=30000)
    custom_stop_callback = CustomStopCallback(eval_env, reward_threshold=300, target_difficulty=1.0, verbose=1)
    eval_callback = EvalCallback(
    eval_env,
     callback_on_new_best=custom_stop_callback,
    best_model_save_path=best_model_dir,  # 儲存目錄
    log_path=log_dir,                     # 評估 log
    eval_freq=10000,                      # 每多少步做一次評估
    n_eval_episodes=5,                    # 每次評估用幾次 episode
    deterministic=True,
    render=False
)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=best_model_dir,
        name_prefix="ppo_checkpoint-hard1"
    )
    auto_difficulty_callback = AutoDifficultyCallback(
        eval_env,None, eval_freq=20_000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=shared_flags,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[ curriculum_callback,auto_difficulty_callback,checkpoint_callback,eval_callback],
        tb_log_name=f"ppo_bipedalwalker_custom-{run_id}"
    )

    model.save(os.path.join(best_model_dir, f"ppo_bipedalwalker_custom_final{run_id}"))
    print(f"[run {run_id}] ✅ 訓練完成，已儲存：")
import re
import os

def numeric_full_shorten(fullpath):
    base = os.path.splitext(os.path.basename(fullpath))[0]

    parts = base.split("_x_")   # 依照交配切段
    blocks = []

    for p in parts:
        # 抓 child_genN → N
        gens = re.findall(r"gen(\d+)", p)
        # 抓 modelM → M
        models = re.findall(r"model(\d+)", p)

        block_nums = gens + models  # 按順序接起來

        if len(block_nums) == 0:
            blocks.append("U")
        elif len(block_nums) == 1:
            blocks.append(block_nums[0])
        else:
            blocks.append("_".join(block_nums))

    # 最終用 X 接起來
    return "X".join(blocks)
def more_train():
    mp.set_start_method("spawn",force=True)
    NUM_RUNS=4
    TOTAL_TIMESTEPTS =500_000_000_0
    base_configs =[]
    for run_id in range(9,NUM_RUNS +10):
        base_configs.append(dict(
            run_id=run_id,
            env_id="BipedalWalkerCustom-v0",
            num_envs=8,                 # 依 CPU 調整；你之前寫 256 很吃 RAM 與 CPU，建議先 16~64
            total_timesteps= TOTAL_TIMESTEPTS,
            difficulty=0.0 ,  # 範例：不同 run 用不同起始難度
            base_seed=random.randint(0, 2**31-1) * run_id, # 每個 run 不同 seed
            learning_rate=2.5e-4,
            n_steps=1024,                 # 多進程並跑建議比單跑小一點，避免顯存爆
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            use_subproc=True,
            #resume_path 可填入各自 run 的舊 checkpoint
            #resume_path= "./best_model/run_1/ppo_checkpoint_run1_8000000_steps.zip"
        ))
    max_workers = min(NUM_RUNS,os.cpu_count()or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(train, base_configs))

def test():
    diff=1.0
    for i in range (10):
    # 測試也要用 custom 
        test_seed = 223913247
        #test_seed = 1451341367
        print(diff)
        # env =make_vec_env("BipedalWalkerCustom-v0",num_envs=1, difficulty=diff,base_seed=test_seed,use_subproc=False)
        env = gym.make("BipedalWalkerCustom-v0", difficulty=diff, render_mode="human")
        # model = PPO.load("./best_model/model2.zip",env=env)
        with open(r"C:\BipedalWalker-v3_ppo\BipedalWalker-v3_ppo\best_model\gen7\gen6_ind46.pkl", "rb") as f:
            model = cloudpickle.load(f)
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

        print(f"📊 模型總參數：{total_params:,}")
        print(f"🛠️ 可訓練參數：{trainable_params:,}")
        # model.set_env(vec_env)
        #print(model.policy)          # 一行就能看到主要子模組
        #inspect_crosstalk_weights(model.policy)
        # print("\n*** 正在執行消融操作：將交叉通道權重歸零... ***")
        # zero_initialize_crosstalk(model.policy)
        # print("*** 消融操作完成！模型現在只依靠純淨通道工作。 ***")
        
        # # --- 再次檢查權重，確認已歸零 (可選) ---
        # print("\n--- 檢查【歸零後】的交叉通道權重 ---")
        # inspect_crosstalk_weights(model.policy)
        obs, info = env.reset(seed=test_seed)
        # obs, info = env.reset()
        print("info =", info)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    
    print(f"✅ 測試總回報：{total_reward:.2f}")
    
    env.close()
def inspect_crosstalk_weights(policy: nn.Module):
    """
    檢查並打印模型中交叉通道權重的學習情況。
    """
    print("\n--- 交叉通道權重學習情況審查 ---")
    
    total_cross_weights = 0
    learned_cross_weights = 0
    
    with torch.no_grad():
        for name, module in policy.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                in_half = W.shape[1] // 2
                out_half = W.shape[0] // 2

                if in_half == 0 or out_half == 0:
                    continue

                # 選中右上角 (母 -> 父) 和 左下角 (父 -> 母) 區塊
                cross_mf = W[:out_half, in_half:]
                cross_fm = W[out_half:, :in_half]

                # 計算這兩個區塊權重的平均絕對值
                mean_abs_mf = torch.abs(cross_mf).mean()
                mean_abs_fm = torch.abs(cross_fm).mean()
                
                print(f"層 '{name}':")
                print(f"  - 母->父 通道權重平均絕對值: {mean_abs_mf:.8f}")
                print(f"  - 父->母 通道權重平均絕對值: {mean_abs_fm:.8f}")

                # 統計非零權重數量
                total_cross_weights += cross_mf.numel() + cross_fm.numel()
                learned_cross_weights += torch.count_nonzero(cross_mf) + torch.count_nonzero(cross_fm)

    if total_cross_weights > 0:
        learning_ratio = (learned_cross_weights / total_cross_weights) * 100
        print(f"\n總結：共有 {learned_cross_weights} / {total_cross_weights} ({learning_ratio:.2f}%) 的交叉權重已從0開始學習。")
    else:
        print("\n總結：模型中沒有可檢查的交叉通道。")
def prune_hardseeds():
    import json
    # 載入 hard_seeds
    hardseed_path="./logs/hard_seeds.json"
    model_path="./best_model/ppo_checkpoint-hard6_8000000_steps.zip"
    env_id="BipedalWalkerCustom-v0"
    eval_n=1
    reward_threshold=250
    difficulty_list=None
    with open(hardseed_path, "r") as f:
        data = json.load(f)

    # 2. 轉新格式（如果是 list 就轉成 {seed: 0}）
    new_data = {}
    for diff, seeds in data.items():
        if isinstance(seeds, list):
            new_data[diff] = {str(s): 0 for s in seeds}
        else:
            new_data[diff] = seeds

    # 3. 載入模型
    model = PPO.load(model_path)

    # 4. 依序對每個難度測試所有 seed
    pruned_data = {}
    diffs = difficulty_list or sorted([float(k) for k in new_data.keys()])
    for diff in diffs:
        diff_str = str(diff)
        seed_dict = new_data.get(diff_str, {})
        if not seed_dict:
            continue
        print(f"\n==== 難度 {diff_str} ====")
        env = gym.make(env_id, difficulty=diff)
        new_seed_dict = {}
        for seed_str in list(seed_dict.keys()):
            rewards = []
            for _ in range(eval_n):
                seedtest=int(seed_str)
                obs, info = env.reset(seed=seedtest)
                print("info =", info)
                done = False
                total_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                rewards.append(total_reward)
            avg_reward = sum(rewards) / len(rewards)
            print(f"Seed {seed_str} 平均回報：{avg_reward:.2f}", end=" ")
            # 只保留沒過關的 seed
            if avg_reward <= reward_threshold:
                new_seed_dict[seed_str] = avg_reward
                print("[保留]")
            else:
                print("[移除]")
        if new_seed_dict:
            pruned_data[diff_str] = new_seed_dict

    # 5. 覆寫 hard_seeds.json
    with open(hardseed_path, "w") as f:
        json.dump(pruned_data, f, indent=2)
    print("\n已完成所有 seed 測試，hard_seeds.json 已更新！")
def rename_modelname():
    model_dir = "./best_model/"
    files = sorted([f for f in os.listdir(model_dir) if f.endswith(".zip")])

    for i, filename in enumerate(files):
        new_name = f"{i}.zip"
        old_path = os.path.join(model_dir, filename)
        new_path = os.path.join(model_dir, new_name)
        os.rename(old_path, new_path)
        print(f"✅ 已將 {filename} ➜ {new_name}")
import json
import os

LINEAGE_FILE = "./lineage.json"

# 如果 lineage.json 不存在，就建立一個空的
if not os.path.exists(LINEAGE_FILE):
    with open(LINEAGE_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)


def save_lineage(child_name, generation, child_id, dad_name, mom_name):
    with open(LINEAGE_FILE, "r", encoding="utf-8") as f:
        lineage = json.load(f)

    lineage[child_name] = {
        "generation": generation,
        "id": child_id,
        "parents": [dad_name, mom_name]
    }

    with open(LINEAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)

def sample_hardseeds(hardseed_path, num_samples=100, diff_min=0.1, diff_max=1.0):
    """ 從 hard_seeds.json 抽 (difficulty, seed) 並可限難度範圍 """
    with open(hardseed_path, "r") as f:
        hardseed_dict = json.load(f)

    all_pairs = []
    for diff_str, seeds in hardseed_dict.items():
        d = float(diff_str)
        if diff_min <= d <= diff_max:
            for seed_str in seeds.keys():
                all_pairs.append((d, int(seed_str)))
    return random.sample(all_pairs, min(num_samples, len(all_pairs)))
TEACHERS = None
DEVICE = None
def _init_worker(model_paths, device_str):
    """
    初始化子程序時載入所有模型到記憶體中。
    支援 .zip (SB3 格式) 和 .pkl (cloudpickle 儲存的 policy)。
    若同名模型同時存在兩種格式，優先讀取 .pkl。
    """
    print("\n[DEBUG] Received model_paths:")
    for p in model_paths:
        print(" -", p, "exists:", os.path.exists(p))

    global TEACHERS, DEVICE
    DEVICE = device_str
    TEACHERS = {}

    # ✅ 讓 .pkl 優先於 .zip
    model_paths = sorted(model_paths, key=lambda x: (not x.endswith(".pkl"), x))

    for p in model_paths:
        name = os.path.basename(p)
        try:
            if p.endswith(".pkl"):
                with open(p, "rb") as f:
                    policy = cloudpickle.load(f)
                dummy_env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)
                model = PPO("MlpPolicy", dummy_env, verbose=0)
                model.policy = policy.to(model.device)
                print(f"✅ Loaded .pkl model: {name}")
            elif p.endswith(".zip"):
                model = PPO.load(p, device=DEVICE)
                print(f"✅ Loaded .zip model: {name}")
            else:
                raise ValueError(f"Unknown file type: {p}")
            TEACHERS[name] = model
        except Exception as e:
            print(f"[❌] Failed to load {p}: {e}")
@torch.inference_mode()
def _get_gaussian_params_step(model, obs_tensor):
    """
    單步：obs_tensor shape [1, obs_dim] => 回傳 mean[act_dim], log_std[act_dim]
    """
    policy = model.policy
    obs_tensor = obs_tensor.to(next(policy.parameters()).device)

    # 走標準路徑取 mean
    feats = policy.extract_features(obs_tensor)
    if policy.share_features_extractor:
        latent_pi, _ = policy.mlp_extractor(feats)
    else:
        latent_pi, _ = policy.mlp_extractor(feats, None)
    mean = policy.action_net(latent_pi)  # [1, act_dim]

    # 取得 log_std
    if hasattr(policy, "log_std") and policy.log_std is not None:
        log_std = policy.log_std.expand_as(mean)
    else:
        dist = policy.get_distribution(obs_tensor)
        if hasattr(dist, "log_std") and dist.log_std is not None:
            log_std = dist.log_std
        elif hasattr(dist, "distribution"):
            std = dist.distribution.stddev
            log_std = torch.log(std + 1e-8)
        else:
            raise RuntimeError("找不到 log_std，請檢查 SB3 版本")

    return mean.squeeze(0), log_std.squeeze(0)


@torch.inference_mode()
def _get_gaussian_params_batch(model, obs_batch):
    """
    批量：obs_batch shape [T, obs_dim] => 回傳 mean[T, act_dim], log_std[T, act_dim]
    """
    policy = model.policy
    device = next(policy.parameters()).device
    X = obs_batch.to(device)

    # 先算 mean（自帶 batch 維）
    feats = policy.extract_features(X)
    if policy.share_features_extractor:
        latent_pi, latent_vl = policy.mlp_extractor(feats)
    else:
        latent_pi,latent_vl = policy.mlp_extractor(feats, None)
    mean = policy.action_net(latent_pi)  # [T, act_dim]
    value =policy.value_net(latent_vl)
    # 取 log_std：優先使用 policy.log_std；否則從 distribution 取
    if hasattr(policy, "log_std") and policy.log_std is not None:
        log_std = policy.log_std.expand_as(mean)  # [T, act_dim]
    else:
        dist = policy.get_distribution(X)  # 會重新抽 features，但穩妥
        if hasattr(dist, "log_std") and dist.log_std is not None:
            log_std = dist.log_std
        elif hasattr(dist, "distribution"):
            std = dist.distribution.stddev
            log_std = torch.log(std + 1e-8)
        else:
            raise RuntimeError("找不到 log_std，請檢查 SB3 版本")

    return mean, log_std,value


# ========== Stage-1：只算 fitness + 輕量存 states ==========
def run_fitness_task(args):
    """
    args: (controller_name, diff, seed_list, env_id, eval_n, reward_threshold, tmp_dir)
    回傳: (controller_name, fitness_dict, tmp_paths)
    """
    (ctrl_name, diff, seed_list,
     env_id, eval_n, reward_th, tmp_dir) = args

    ctrl = TEACHERS[ctrl_name]
    env  = gym.make(env_id, difficulty=diff)

    tot_r = eps = passed = 0
    tmp_paths = []
    rewards = []

    for seed in seed_list:                    # ← 多個 seed
        for ep in range(eval_n):
            obs, _ = env.reset(seed=to_uint32(seed))
            done, ep_r = False, 0.0
            traj = []

            while not done:
                obs_t = (torch.as_tensor(obs, dtype=torch.float32)
                           .unsqueeze(0).to(DEVICE))
                mean, _ = _get_gaussian_params_step(ctrl, obs_t)
                action  = mean.detach().cpu().numpy()

                traj.append(np.array(obs, copy=True))
                obs, r, term, trunc, _ = env.step(action)
                ep_r += r; done = term or trunc

            tot_r += ep_r; eps += 1
            if ep_r >= reward_th: passed += 1
            rewards.append(ep_r)
            arr = np.asarray(traj, dtype=np.float16)
            os.makedirs(tmp_dir, exist_ok=True)
            short_hash = hashlib.md5(ctrl_name.encode()).hexdigest()[:8]
            fn = os.path.join(
                tmp_dir,
                f"{short_hash}__d{diff:.2f}__s{seed}__p{os.getpid()}__e{ep}.npz"
            )
            np.savez_compressed(
                fn,
                states=arr,
                diff=np.float32(diff),
                seed=np.int64(seed)
            )
            tmp_paths.append(fn)

    env.close()
    return ctrl_name, {
    "total_reward": tot_r,
    "episodes": eps,
    "passed": passed,
    "rewards": rewards
}, tmp_paths

def to_uint32(x: int) -> int:
    """将任意 Python int 映射到 32-bit 无符号整型范围，适合传给 env.reset(seed=...)."""
    return int(x) & 0xFFFFFFFF
# ========== Stage-2：離線對 Top-K 批量標註（不重跑環境） ==========
def offline_label_task(args):
    """
    args: (states_path, top_teacher_names)
    讀回 states，對 top_teacher 做批量前向 => 產生 label 檔（.label.npz）
    回傳: label_path
    """
    states_path, top_teacher_names = args
    data = np.load(states_path)
    S = data["states"].astype(np.float32)            # [T, obs_dim]
    diff = float(data["diff"])
    seed = int(data["seed"])
    T = S.shape[0]

    obs_batch = torch.from_numpy(S)                  # [T, obs_dim]
    means_list, logstds_list, actions_list, confs_list,value_list = [], [], [], [],[]

    with torch.inference_mode():
    # 先一次跑完所有老師，收齊 mean/logstd/value
        tmp = []
        for name in top_teacher_names:
            m, ls, v = _get_gaussian_params_batch(TEACHERS[name], obs_batch)  # [T, A], [T, A], [T, 1]
            tmp.append((
                m.detach().cpu().numpy(),       # [T, A]
                ls.detach().cpu().numpy(),      # [T, A] (log_std)
                v.detach().cpu().numpy().squeeze(-1)  # [T]
            ))

        # ====== Epistemic：老師間均值一致性 ======
        means_stack = np.stack([x[0] for x in tmp], axis=0)   # [Tchr, T, A]
        means_stack = np.transpose(means_stack, (1, 0, 2))    # [T, Tchr, A]
        inter_teacher_var = np.var(means_stack, axis=1)       # [T, A]
        agree_conf = 1.0 / (np.mean(inter_teacher_var, axis=1) + 1e-8)  # [T]
        # 簡單的 5~95 去尾 + [0,1] 歸一化
        def minmax_norm(x, eps=1e-8):
            lo, hi = np.percentile(x, 5), np.percentile(x, 95)
            x = np.clip(x, lo, hi)
            x = (x - lo) / (hi - lo + eps)
            return np.clip(x, 0.0, 1.0)
        agree_conf = minmax_norm(agree_conf)  # [T] in [0,1]

        # ====== 依序產生每位老師的 conf，含 value 門控 ======
        alpha = 0.6  # aleatoric vs epistemic 幾何融合權重
        k     = 0.5  # value 門控斜率
        for (m_cpu, ls_cpu, v_cpu) in tmp:
            # Aleatoric：反方差（對角高斯）
            var_cpu = np.exp(2.0 * ls_cpu)                       # [T, A]
            alea_conf = 1.0 / (np.mean(var_cpu, axis=1) + 1e-8)  # [T]
            alea_conf = minmax_norm(alea_conf)

            # 幾何融合（aleatoric × epistemic）
            conf_cpu = np.exp(alpha * np.log(alea_conf + 1e-12) +
                            (1 - alpha) * np.log(agree_conf + 1e-12))  # [T]

            # ====== Value 門控（越高價值 → 權重越大；你也可改成反向）======
            # 先把 value 做穩定化：去尾 + z-score + sigmoid 門控
            v_clip_lo, v_clip_hi = np.percentile(v_cpu, 5), np.percentile(v_cpu, 95)
            v_clip = np.clip(v_cpu, v_clip_lo, v_clip_hi)
            v_mu, v_std = np.mean(v_clip), (np.std(v_clip) + 1e-8)
            v_z = (v_clip - v_mu) / v_std
            gate = 1.0 / (1.0 + np.exp(-k * v_z))  # [T] in (0,1)
            conf_cpu *= gate
            conf_cpu = minmax_norm(conf_cpu)

            # 你原本就要存的資料
            a_cpu = m_cpu.copy()               # 以 mean 作為動作標註
            means_list.append(m_cpu)
            logstds_list.append(ls_cpu)
            actions_list.append(a_cpu)
            value_list.append(v_cpu)           # 仍保存每位老師自己的 value
            confs_list.append(conf_cpu)

    # 轉為 [T, Tchr, ...] 方便後續 concat（沿 T 維拼）
    Tchr = len(top_teacher_names)
    M = np.stack(means_list, axis=0).transpose(1, 0, 2)     # [T, Tchr, act_dim]
    LS = np.stack(logstds_list, axis=0).transpose(1, 0, 2)  # [T, Tchr, act_dim]
    A = np.stack(actions_list, axis=0).transpose(1, 0, 2)   # [T, Tchr, act_dim]
    C = np.stack(confs_list, axis=0).transpose(1, 0)        # [T, Tchr]
    V = np.stack(value_list, axis=0).transpose(1, 0)
    out = states_path.replace(".npz", ".label.npz")
    np.savez_compressed(out,
                        teacher_means=M.astype(np.float32),
                        teacher_logstds=LS.astype(np.float32),
                        teacher_actions=A.astype(np.float32),
                        confidences=C.astype(np.float32),
                        teacher_values=V.astype(np.float32),
                        diff=np.float32(diff),
                        seed=np.int64(seed))
    return out


# ========== 主流程：先 fitness 後收集 Top-K ==========
def test_population(
    model_paths,
    hardseed_path,
    env_id="BipedalWalkerCustom-v0",
    num_samples=300,
    eval_n_stage1=1,
    eval_n_stage2=10,
    reward_threshold=250,
    top_k=30,
    save_dir="./logs/ga_eval",
    save_name="mtkd_continuous.pt",
    num_workers=12,
    device_str="cpu",
    diff_min=0.0,
    diff_max=1.0,
    generate_distill=True, 
    fixed_pairs=None,
):
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = os.path.join(save_dir, "tmp_states"); os.makedirs(tmp_dir, exist_ok=True)
    teacher_names = [os.path.basename(p) for p in model_paths]

    # 取樣 (diff, seed) ------------- 第一層批量切段 ----------------
    PAIR_BATCH = 50                      # 一段最多 100 對 (diff, seed)
    if fixed_pairs is not None:
        print("📌 使用外部提供的固定 seed 組（所有世代一致）")
        sampled_pairs = fixed_pairs
    else:
        sampled_pairs = sample_hardseeds(hardseed_path, num_samples,
                                        diff_min=diff_min, diff_max=diff_max)
    sampled_pairs.sort()                   # optional: 讓同 diff 靠近

    pair_batches = [
        sampled_pairs[i : i + PAIR_BATCH]
        for i in range(0, len(sampled_pairs), PAIR_BATCH)
    ]
    print(f"Total (diff, seed) pairs = {len(sampled_pairs)}, "
          f"{len(pair_batches)} batches × {PAIR_BATCH} (<=) each")

    # 累積器
    fitness = {n: {"total_reward":0., "episodes":0, "passed":0, "rewards": []} for n in teacher_names}
    paths_map = {n: [] for n in teacher_names}

    # --------- 逐段處理 Stage-1 (fitness) ----------
    SEED_BATCH = 10                        # 一個 Pool task 跑 10 seeds
    for b_idx, pair_batch in enumerate(pair_batches, 1):
        print(f"\n🔹 Batch {b_idx}/{len(pair_batches)} : pairs={len(pair_batch)}")

        # 把該段 pairs -> diff → seeds 映射
        diff2seeds = defaultdict(list)
        for diff, seed in pair_batch:
            diff2seeds[diff].append(seed)

        # 針對這段重新建立 tasks1
        tasks1 = []
        for ctrl in teacher_names:
            for diff, seed_list in diff2seeds.items():
                for i in range(0, len(seed_list), SEED_BATCH):
                    batch = seed_list[i : i + SEED_BATCH]
                    tasks1.append((ctrl, diff, batch,
                                   env_id, eval_n_stage1,
                                   reward_threshold, tmp_dir))

        print(f"    ↳ Stage-1 tasks this batch = {len(tasks1)}")
        chunks1 = max(1, len(tasks1) // (num_workers * 4))

        with Pool(processes=num_workers,
                  initializer=_init_worker,
                  initargs=(model_paths, device_str)) as pool:

            for ctrl, fit_res, paths in tqdm(
                    pool.imap_unordered(run_fitness_task, tasks1, chunksize=chunks1),
                    total=len(tasks1), desc="    Stage-1"):
                f = fitness[ctrl]
                f["total_reward"] += fit_res["total_reward"]
                f["episodes"]     += fit_res["episodes"]
                f["passed"]       += fit_res["passed"]
                if "rewards" in fit_res:
                     f["rewards"].extend(fit_res["rewards"])
                paths_map[ctrl].extend(paths)

        # ----- 釋放 Pool / env / GPU 記憶體，再進下一段 -----

    # --------- Top-K 挑選 ----------
    fitness_out = {}
    for n, f in fitness.items():
        rewards = np.array(f["rewards"]) if len(f["rewards"]) > 0 else np.array([f["total_reward"]/max(1,f["episodes"])])
        avg = float(np.mean(rewards))
        std = float(np.std(rewards))
        fitness_out[n] = {
            "avg_reward": avg,
            "std_reward": std,
            "passed": f["passed"],
            "episodes": f["episodes"]
        }
    with open(os.path.join(save_dir, "fitness_stage1.json"), "w", encoding="utf-8") as fj:
        json.dump(fitness_out, fj, indent=2, ensure_ascii=False)

    top_names = sorted(fitness_out,
                       key=lambda n: fitness_out[n]["avg_reward"],
                       reverse=True)[:min(top_k, len(teacher_names))]
    print("\nTop-K controllers:")
    for name in top_names[:5]:
        stats = fitness_out[name]
        print(f"  {name:<40} → avg={stats['avg_reward']:.2f}, std={stats['std_reward']:.2f}")
    fitness_top = {name: fitness_out[name] for name in top_names}

# 2. 將這個只包含 Top-K 成績的字典，存成新的 JSON 檔案
    top_k_filename = os.path.join(save_dir, "fitness_top.json")
    with open(top_k_filename, "w", encoding="utf-8") as fj:
        json.dump(fitness_top, fj, indent=2, ensure_ascii=False)

    print(f"Top-{len(top_names)} fitness scores saved to {top_k_filename}")
    if not generate_distill:
        print("\n⚙️ Skipped Stage-2 distillation dataset generation.")
        print("✅ Done (fitness only).")
        return fitness_out, top_names
    # --------- Stage-2 (offline label) ----------
    label_tasks = [
        (p, top_names) for ctrl in top_names for p in paths_map[ctrl]
    ]
    chunks2 = max(1, len(label_tasks) // (num_workers * 4))

    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(model_paths, "cuda")) as pool:
        list(tqdm(pool.imap_unordered(offline_label_task, label_tasks, chunksize=chunks2),
                  total=len(label_tasks), desc="Stage-2 label"))

    # --------- 合併並存檔（與你原本相同） ----------
    S,M,LS,A,C,D,Z,V = [],[],[],[],[],[],[],[]
    for lp in tqdm([p.replace(".npz", ".label.npz") for p,_ in label_tasks], desc="Merging"):
        sp = lp.replace(".label.npz", ".npz")
        s  = np.load(sp); l = np.load(lp)
        S.append(s["states"].astype(np.float32))
        D.append(np.full((S[-1].shape[0],), float(s["diff"]), np.float32))
        Z.append(np.full((S[-1].shape[0],), int(np.int64(s["seed"])), np.int64))
        M.append(l["teacher_means"]); LS.append(l["teacher_logstds"])
        A.append(l["teacher_actions"]); C.append(l["confidences"])
        V.append(l["teacher_values"])   
    torch.save({
        "states": torch.from_numpy(np.concatenate(S,0)),
        "teacher_means":   torch.from_numpy(np.concatenate(M,0)),
        "teacher_logstds": torch.from_numpy(np.concatenate(LS,0)),
        "teacher_actions": torch.from_numpy(np.concatenate(A,0)),
        "confidences":     torch.from_numpy(np.concatenate(C,0)),
        "teacher_names": top_names,
        "difficulties": torch.from_numpy(np.concatenate(D,0)),
        "seeds":        torch.from_numpy(np.concatenate(Z,0)).long(),
        "teacher_values":  torch.from_numpy(np.concatenate(V,0)),
    }, os.path.join(save_dir, save_name))

    print(f"\n✅ Done. steps={sum(x.shape[0] for x in S)}, Top-K={len(top_names)}")
    return fitness_out, top_names
def _index_get(mod,key):
    if key.isdigit():#是否包含數字
        idx = int(key)
        if isinstance(mod,(nn.Sequential,nn.ModuleList,list,tuple)):
            return mod[idx]
        raise TypeError(f"{type(mod)}不適可索引容器:{key}")
    if isinstance(mod,nn.ModuleDict):
        if key in mod:return mod[key]
        raise KeyError(f"ModuelDict無鍵:{key}")
    if hasattr(mod,key):
        return getattr(mod,key)
    raise AttributeError(f"{type(mod)} 無屬性：{key}")
def test_all_generations(
    root="./generations",
    hardseed_path="./logs/hard_seeds.json",
    env_id="BipedalWalkerCustom-v0",
    num_samples=1000,
    num_workers=10,
    device_str="cpu",
):
    # 🔥 Step 1：抽一次種子（全域固定）
    fixed_pairs = sample_hardseeds(
        hardseed_path,
        num_samples=num_samples,
        diff_min=0.0,
        diff_max=1.0
    )

    print(f"📌 固定抽樣 {len(fixed_pairs)} 個 (difficulty, seed) 配對，所有世代將使用相同測試集")

    # 🔥 Step 2：逐世代測試（每代都用 fixed_pairs）
    gens = sorted(os.listdir(root))
    gens = [g for g in gens if os.path.isdir(os.path.join(root, g))]

    for gen in gens:
        gen_dir = os.path.join(root, gen)
        print(f"\n==============================")
        print(f"  🔥 測試世代: {gen}")
        print(f"==============================")

        model_files = sorted([
            os.path.join(gen_dir, f)
            for f in os.listdir(gen_dir)
            if f.endswith(".zip") or f.endswith(".pkl")
        ])
        if len(model_files) == 0:
            print(f"⚠️ {gen} 無模型，跳過")
            continue

        save_dir = f"./logs/ga_eval_{gen}"
        os.makedirs(save_dir, exist_ok=True)

        # ⭐ 用固定種子跑 test_population
        fitness_out, top_names = test_population(
            model_paths=model_files,
            hardseed_path=hardseed_path,
            env_id=env_id,
            num_samples=num_samples,
            eval_n_stage1=1,
            eval_n_stage2=10,
            reward_threshold=250,
            top_k=10,
            save_dir=save_dir,
            save_name=f"distill_{gen}.pt",
            num_workers=num_workers,
            device_str=device_str,
            diff_min=0.0,
            diff_max=1.0,
            generate_distill=False,
            fixed_pairs=fixed_pairs,   # ⭐ 關鍵
        )

        print(f"🎯 {gen} 測試完成 → JSON 存在 {save_dir}")
def _index_set(mod, key, new):
    """對 Sequential/ModuleList/ModuleDict 設定元素；否則用 setattr。"""
    if key.isdigit():
        idx = int(key)
        if isinstance(mod, (nn.Sequential, nn.ModuleList, list)):
            mod[idx] = new
            return
        raise TypeError(f"當前模組 {type(mod)} 不是可索引容器，卻用數字索引 {key}")
    if isinstance(mod, nn.ModuleDict):
        mod[key] = new
        return
    setattr(mod, key, new)
def resolve_parent_and_key(root, path: str):
    """回傳 (parent_module, last_key)。"""
    parts = path.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[:-1]:
        parent = _index_get(parent, p)
    return parent, parts[-1]
def get_module_by_path(root, path: str):
    mod = root
    for p in path.split("."):
        mod = _index_get(mod, p)
    return mod
def set_module_by_path(root, path: str, new_layer: nn.Module):
    """
    把路徑指定的那一層替換成 new_layer。
    例：set_module_by_path(model.policy, "mlp_extractor.policy_net.0", nn.Linear(24,64))
    """
    parent, last = resolve_parent_and_key(root, path)

    # 讓新層的 device/dtype 盡量和舊層一致（避免不小心留在 CPU）
    try:
        old = _index_get(parent, last)
        # 若舊層有參數，就用它的第一個參數的 device/dtype 對齊
        for p in old.parameters():
            new_layer = new_layer.to(device=p.device, dtype=p.dtype)
            break
    except Exception:
        pass  # 對齊失敗也沒關係，不阻塞替換

    _index_set(parent, last, new_layer)
def is_param_layer(m:nn.Module) -> bool:
    return isinstance(m,(nn.Linear,nn.Conv2d))
def param_layer_indices(seq: nn.Sequential):
    return [i for i,m in enumerate(seq) if is_param_layer(m)]
@dataclass
class Gene:
    seq_path: str
    idx: int
    kind: str
    shape: tuple
def collect_genes(policy) -> List[Gene]:
    genes: List[Gene] = []
    seq_paths = [
        "mlp_extractor.policy_net",   # Actor 隱藏
        "mlp_extractor.value_net",    # Critic 隱藏
    ]
    # 頭部（非 Sequential）：當作單獨基因（不插適應層）
    head_paths = ["action_net", "value_net"]
    for sp in seq_paths:
        seq = get_module_by_path(policy, sp) 
        for i in param_layer_indices(seq):
            m =seq[i]
            kind = "linear" if isinstance(m,nn.Linear)else"conv2d"
            genes.append(Gene(seq_path=sp,idx=i,kind=kind,shape=tuple(m.weight.shape)))
        for hp in head_paths:
            m = getattr(policy, hp)
            kind = "linear" if isinstance(m, nn.Linear) else "conv2d"
            genes.append(Gene(seq_path=hp, idx=-1, kind=kind, shape=tuple(m.weight.shape)))
    return genes
class ScaleShift(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
    def forward(self,x):return self.gamma*x+self.beta
class BoundaryAdapter(nn.Module):
    def __init__(self,core:nn.Module,kind:str,in_dim: int,out_dim: int):
        super().__init__()
        self.core ,self.kind,self.in_dim,self.out_dim=core,kind,in_dim,out_dim
        self.is_adapter=True
    def forward(self,x): return self.core(x)
def make__adapter(prev: nn.Module,nxt:nn.Module)->nn.Module:
    if isinstance(prev,nn.Linear) and isinstance(nxt,nn.Linear):
        in_d,out_d =prev.out_features,nxt.in_features
        if in_d==out_d:
            return BoundaryAdapter(ScaleShift((out_d,)),"scaleshift",in_d,out_d)
        core = nn.Linear (in_d,out_d,bias=True)
        nn.init.orthogonal_(core.weight);nn.init.zeros_(core.bias)
        return BoundaryAdapter(core,"linear",in_d,out_d)
    if isinstance(prev, nn.Conv2d) and isinstance(nxt, nn.Conv2d):
        in_c, out_c = prev.out_channels, nxt.in_channels
        if in_c == out_c:
            return BoundaryAdapter(ScaleShift((out_c,1,1)), "scaleshift", in_c, out_c)
        core = nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(core.weight, a=math.sqrt(5)); nn.init.zeros_(core.bias)
        return BoundaryAdapter(core, "conv1x1", in_c, out_c)
    raise NotImplementedError("如需 Conv↔Linear（含 Flatten）可再加 probe。")

@torch.no_grad()
def copy_params(dst:nn.Module,src:nn.Module):
    if hasattr(dst,"weight") and hasattr(src,"weight") and dst.weight.shape ==  src.weight.shape:
        dst.weight.data.copy_(src.weight.data) 
    if getattr(dst,"bias",None) is not None and getattr(src,"bias",None) is not None:
        if dst.bias.shape == src.bias.shape:
            dst.bias.data.copy_(src.bias.data)
def insert_between(seq: nn.Sequential,idx_prev:int,adapter:nn.Module) -> nn.Sequential:
    layers=[]
    for i ,m in enumerate(seq):
        layers.append(m)
        if i == idx_prev:
            layers.append(adapter)
    return nn.Sequential(*layers)
@torch.no_grad()
def crossover_one_boundary(
    child_policy: nn.Module,
    father_policy: nn.Module,
    mother_policy: nn.Module,
    p_from_mother: float = 0.5,     # 保留介面相容性（本版不使用）
    device: str = "cuda",
    forbidden: Optional[Dict[str, Set[int]]] = None,
    debug_target: Optional[Tuple[str, int]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    每輪只交配一個「單位」：
      - 單位可以是：MLP extractor 的單一參數層，或整個 head(action_net/value_net)
      - 以父為基底，只覆蓋母親到該單位；其餘保持父
      - 若交配的是 MLP 層：在第一個 M/F 邊界插入 1 個 adapter（同維度→ScaleShift）
      - forbidden: {"mlp_extractor.policy_net": {idx1, ...}, "action_net": {-1}, ...}
    回傳 info: switches, adapter, touched
    """
    if forbidden is None:
        forbidden = {}

    BACKBONES = ("mlp_extractor.policy_net", "mlp_extractor.value_net")
    HEADS     = ("action_net", "value_net")  # 注意：與 backbone 的 value_net 不同路徑

    switches: Dict[str, Dict[int, str]] = {}
    touched:  Dict[str, Set[int]] = {}

    # ------- 蒐集候選：MLP 參數層 + 兩個 head（排除 forbidden） -------
    candidates: List[Tuple[str, Optional[int]]] = []

    # MLP 參數層
    for sp in BACKBONES:
        seq = get_module_by_path(child_policy, sp)
        for idx in param_layer_indices(seq):
            if sp in forbidden and idx in forbidden[sp]:
                continue
            candidates.append((sp, idx))

    # head：用 -1 當索引
    for sp in HEADS:
        if sp in forbidden and (-1 in forbidden[sp]):
            continue
        candidates.append((sp, -1))

    # 若沒有可用候選，直接返回
    if not candidates:
        return child_policy, dict(switches=switches, adapter=None, touched=touched)

    # ------- 隨機選「一個」單位交配 -------
    if debug_target and debug_target in candidates:
        # 如果提供了有效的 debug_target，就使用它
        target_sp, target_idx = debug_target
        print(f"[DEBUG PICK] unit={target_sp} idx={target_idx}")
    else:
        # 否則，恢復隨機選擇
        target_sp, target_idx = random.choice(candidates)
        print(f"[RANDOM PICK] unit={target_sp} idx={target_idx}")

    # 如果選到 head：整塊由母親覆蓋，其它不動；不插 adapter
    if target_sp in HEADS:
        dst = getattr(child_policy, target_sp)
        src = getattr(mother_policy, target_sp)
        copy_params(dst, src)
        touched.setdefault(target_sp, set()).add(-1)
        # head 不參與 switches/adapter（不在 backbone 上）
        return child_policy, dict(switches=switches, adapter=None, touched=touched)

    # ------- 若選到 MLP 層：只該層覆蓋母親，其餘標記為父 -------
    # 先把兩條 backbone 的參數層都標記好（選中層 M，其他 F）
    for sp in BACKBONES:
        seq_c   = get_module_by_path(child_policy, sp)
        seq_mom = get_module_by_path(mother_policy, sp)
        idxs    = param_layer_indices(seq_c)

        for idx in idxs:
            if sp == target_sp and idx == target_idx:
                # 覆蓋母親該層
                copy_params(seq_c[idx], seq_mom[idx])
                switches.setdefault(sp, {})[idx] = "M"
                touched.setdefault(sp, set()).add(idx)
            else:
                switches.setdefault(sp, {})[idx] = "F"  # 保持父親（child 已是父 clone，無需 copy）

    # ------- 在第一個 M/F 邊界插入 1 個 adapter（同維度→ScaleShift） -------
    adapters_info = None
    def _make_scaleshift_adapter(prev_layer: nn.Module, next_layer: nn.Module) -> nn.Module:
        feat = getattr(next_layer, "in_features", None)
        if feat is None:
            feat = getattr(prev_layer, "out_features", None)
        if feat is None:
            raise TypeError(f"無法推斷 ScaleShift 維度，prev={type(prev_layer)}, next={type(next_layer)}")
        ss = ScaleShift((feat,))
        return BoundaryAdapter(ss, "scaleshift", feat, feat)

    for sp in BACKBONES:
        seq  = get_module_by_path(child_policy, sp)
        idxs = param_layer_indices(seq)
        marks = switches.get(sp, {})

        # 找第一個相鄰參數層的 M/F 切換
        for j in range(len(idxs) - 1):
            i_prev, i_next = idxs[j], idxs[j + 1]
            left_src, right_src = marks.get(i_prev), marks.get(i_next)
            if left_src is None or right_src is None:
                continue
            if left_src != right_src:
                insert_pos = i_next - 1
                adp = make__adapter(seq[i_prev], seq[i_next]).to(device)
                # 保險：若工廠回傳 Identity，換成 ScaleShift（通常你的工廠已回 scaleshift）
                if isinstance(adp.core, nn.Identity):
                    adp = _make_scaleshift_adapter(seq[i_prev], seq[i_next]).to(device)
                new_seq = insert_between(seq, insert_pos, adp)
                set_module_by_path(child_policy, sp, new_seq)
                adapters_info = dict(
                    seq_path=sp,
                    between=(i_prev, i_next),
                    adapter=type(adp.core).__name__,
                )
                break
        if adapters_info:
            break  # 本輪只插 1 個
    return child_policy, dict(switches=switches, adapter=adapters_info, touched=touched)
def freeze_except_adapters(model:nn.Module):
    for p in model.parameters(): p.requires_grad =False
    for m in model.modules():
        if getattr(m,"is_adapter",False):
            for p in m.parameters():p.requires_grad=True
def unfreeze_all(model:nn.Module):
    for p in model.parameters(): p.requires_grad =True
def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
            return None # 應該加上找不到的處理
def train_adapters_offline(
    ppo_model: PPO,
    pt_path: str,
    adapter_info: dict, # <--【新增】接收來自 crossover 的 info['adapter']
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 4096,
    lambda_std: float = 0.05,
    device: str = "cuda",
    temperature =0.5 ,
):
    blob = torch.load(pt_path, map_location="cuda")
    states = blob["states"].float()  # [N, obs_dim]
    means  = blob["teacher_means"].float()  # [N,Tchr,act_dim]
    logstds_t = blob["teacher_logstds"].float()
    conf   = blob["confidences"].float()    # [N,Tchr]
    values_t = blob["teacher_values"].float()
    # 加權平均老師 mean 作目標
    # --- 【新的目標準備邏輯：只聽最自信的老師】 ---

# 1. 對於每一個狀態，找到信賴度最高的老師的索引
#    conf 的形狀是 [N, Tchr]，argmax(dim=1) 會返回每個樣本中最大值的索引
    best_teacher_indices = torch.argmax(conf, dim=1)  # 結果形狀為 [N]

    # 2. 準備一個索引器，用於從原始數據中高效地提取數據
    N = states.shape[0]
    # 這是 PyTorch/NumPy 中非常高效的進階索引 (Advanced Indexing)
    idx_gather = torch.arange(N, device=device)

    # 3. 使用這個索引器，從原始數據中只挑選出「最佳老師」的預測作為目標
    mu_tgt = means[idx_gather, best_teacher_indices, :]
    logstd_tgt = logstds_t[idx_gather, best_teacher_indices, :]
    value_tgt = values_t[idx_gather, best_teacher_indices]

# --- 後續程式碼不變 ---
    print("\n" + "="*20 + " Shape Debugging " + "="*20)
    print(f"Loaded 'states' shape:           {states.shape}")
    print(f"Calculated 'mu_tgt' shape:       {mu_tgt.shape}")
    print(f"Calculated 'logstd_tgt' shape:    {logstd_tgt.shape}")
    print(f"Calculated 'value_tgt' shape:      {value_tgt.shape}  <--- 請重點關注這一行！")
    print("="*57 + "\n")
    train_target ="policy" if "policy" in adapter_info['seq_path'] else "critic"
    print(f"->Targeting new adapter in'{adapter_info['seq_path']}'. Training mode: {train_target}")
    if train_target == "policy":
        ds =torch.utils.data.TensorDataset(states,mu_tgt,logstd_tgt)
    else:
        ds =torch.utils.data.TensorDataset(states,value_tgt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = ppo_model.policy.to(device)
    freeze_except_adapters(policy)
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"DEBUG: Found {len(params)} trainable parameters.")
    print(f"DEBUG: Trainable parameter shapes: {[p.shape for p in params]}")
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    def get_gaussian(obs_batch):
        f = policy.extract_features(obs_batch)
        lat_pi, lat_vf = policy.mlp_extractor(f) if policy.share_features_extractor \
                   else policy.mlp_extractor(f, None)
        mu = policy.action_net(lat_pi)
        log_std = policy.log_std.expand_as(mu) if hasattr(policy, "log_std") else torch.zeros_like(mu)
        value = policy.value_net(lat_vf)
        return mu, log_std,value

    policy.train()
    with torch.enable_grad():
        for ep in range(epochs):
            run_l, n = 0.0, 0
            for data_batch in dl:
                data_batch=[t.to(device)for t in data_batch]
                obs =data_batch[0]
                opt.zero_grad(set_to_none=True) # 清除上一輪的梯度
                mu_s,logstds_s,values_s =get_gaussian(obs)
                if train_target=="policy":
                    # mu_t,logstds_t=data_batch[1],data_batch[2]
                    # var_t = torch.exp(2 * logstds_t)
                    # var_s = torch.exp(2 * logstds_s)
                    # if temperature != 1.0:
                        
                    # # 確保溫度是一個張量以便進行廣播運算
                    #     temp_tensor = torch.tensor(temperature, device=device)
                    #     # 調整教師的 logstd 和 variance
                    #     var_t = var_t * (temp_tensor**2)
                    #     logstds_t = logstds_t + torch.log(temp_tensor)
                    # kl_div = (logstds_s - logstds_t) + (var_t + (mu_t - mu_s).pow(2)) / (2 * var_s) - 0.5
                    # loss = kl_div.sum(dim=-1).mean()
                    mu_t, logstds_t = data_batch[1], data_batch[2]
                    loss_mu  = torch.nn.functional.mse_loss(mu_s, mu_t)
                    loss_std = torch.nn.functional.mse_loss(logstds_s, logstds_t)

                    loss = loss_mu 
                else:
                    values_t= data_batch[1]
                    loss=((values_s.squeeze()-values_t)**2).mean()
                loss.backward()
                # if len(params) > 0:
                #     total_norm = 0
                #     for p in params:
                #         if p.grad is not None:
                #             param_norm = p.grad.data.norm(2)
                #             total_norm += param_norm.item() ** 2
                #     total_norm = total_norm ** 0.5
                #     # 在 print loss 前面或後面加上這一行
                #     print(f"  -> Grad Norm: {total_norm:.6f}", end="")
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                run_l += loss.item() * obs.size(0); n += obs.size(0)
            loss_name = "KL_Loss" if train_target == "policy" else "MSE_Loss"
            print(f"  Adapter epoch {ep+1}/{epochs} \t{loss_name}={run_l/n:.6f}")
    policy.eval()
def train_network_offline(
    ppo_model: PPO,
    pt_path: str,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    batch_size: int = 4096,
    vf_coef: float = 0.5,
    device: str = "cuda",
    kl_target_min=20,
    kl_target_max=80,
    base_lr=1e-4,
    
):
    blob = torch.load(pt_path, map_location="cuda")
    states = blob["states"].float()  # [N, obs_dim]
    means  = blob["teacher_means"].float()  # [N,Tchr,act_dim]
    logstds_t = blob["teacher_logstds"].float()
    conf   = blob["confidences"].float()    # [N,Tchr]
    values_t = blob["teacher_values"].float()
    # 加權平均老師 mean 作目標
    # --- 【新的目標準備邏輯：只聽最自信的老師】 ---
    logstd_vals = logstds_t.flatten().cpu().numpy()
    logstd_mean, logstd_std = np.mean(logstd_vals), np.std(logstd_vals)
    conf_std = conf.float().std().item()

    temperature = 1.5 + 0.3 * max(0, (-2.2 - logstd_mean))
    temperature = float(np.clip(temperature, 1.0, 2.3))
    logstd_floor = -2.3 if logstd_std > 0.8 else -2.5
    alpha = 0.6 if conf_std > 0.2 else 0.7
    lr_scale = (1.0 - 0.5 * np.tanh(logstd_std + conf_std))
    lr = base_lr * lr_scale
    lr = float(np.clip(lr, 5e-5, 2e-4))
    print(f"\n🧮 Auto-Heuristic Init: temp={temperature:.2f}, α={alpha:.2f}, floor={logstd_floor:.2f}, lr={lr:.2e}")
# 1. 對於每一個狀態，找到信賴度最高的老師的索引
#    conf 的形狀是 [N, Tchr]，argmax(dim=1) 會返回每個樣本中最大值的索引
    top_idx  = conf.argmax(dim=1) 
                        # [N]
    row = torch.arange(conf.shape[0], device=conf.device)
    mu_tgt     = means[row, top_idx]                       # [N, act_dim]
    logstd_tgt = logstds_t[row, top_idx] 
    value_tgt  = values_t[row, top_idx]                # [N]
# --- 後續程式碼不變 ---
    print("\n" + "="*20 + " Shape Debugging " + "="*20)
    print(f"Loaded 'states' shape:           {states.shape}")
    print(f"Calculated 'mu_tgt' shape:       {mu_tgt.shape}")
    print(f"Calculated 'logstd_tgt' shape:    {logstd_tgt.shape}")
    print(f"Calculated 'value_tgt' shape:      {value_tgt.shape}  <--- 請重點關注這一行！")
    print("="*57 + "\n")
    logstd_tgt = torch.clamp(logstd_tgt, min=-2.5, max=1.0)  
    ds =torch.utils.data.TensorDataset(states, mu_tgt, logstd_tgt,value_tgt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = ppo_model.policy.to(device)
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"DEBUG: Found {len(params)} trainable parameters.")
    print(f"DEBUG: Trainable parameter shapes: {[p.shape for p in params]}")
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    frozen    = [n for n, p in policy.named_parameters() if not p.requires_grad]
    print(f"🔸 Trainable layers ({len(trainable)}): {trainable[:5]} ...")
    print(f"🔸 Frozen layers ({len(frozen)}): {frozen[:5]} ...")
    #opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    def get_gaussian(obs_batch):
        f = policy.extract_features(obs_batch)
        lat_pi, lat_vf = policy.mlp_extractor(f) if policy.share_features_extractor \
                   else policy.mlp_extractor(f, None)
        mu = policy.action_net(lat_pi)
        log_std = policy.log_std.expand_as(mu) if hasattr(policy, "log_std") else torch.zeros_like(mu)
        value = policy.value_net(lat_vf)
        return mu, log_std,value

    policy.train()
    with torch.enable_grad():
        for ep in range(epochs):
            run_pi_l, run_vf_l,n = 0.0, 0.0,0
            for data_batch in dl:
                data_batch=[t.to(device)for t in data_batch]
                obs, mu_t, logstds_t, values_t = data_batch # 解包所有數據
                optimizer.zero_grad(set_to_none=True) # 清除上一輪的梯度
                mu_s,logstds_s,values_s =get_gaussian(obs)
                
                 # 1. 計算學生和老師的 variance (方差)
                var_s = torch.exp(2 * logstds_s)
                logstds_t_soft = logstds_t
                # 2. 根據溫度調整老師的目標分佈
                if temperature > 1.0:
                    temp_tensor = torch.tensor(temperature, device=device)
                    # 溫度越高，老師的 log_std 越大，分佈越寬
                    logstds_t_soft = logstds_t + torch.log(temp_tensor)
                
                var_t = torch.exp(2 * logstds_t_soft)

                # 3. 計算 KL 散度 KL(P_teacher || P_student)
                # 這是衡量兩個高斯分佈差異的標準公式
                kl_div = (logstds_s - logstds_t_soft) + (var_t + (mu_t - mu_s).pow(2)) / (2 * var_s) - 0.5
                
                # Policy 損失：對 KL 散度在動作維度上求和，然後在 batch 維度上取平均
                loss_soft = kl_div.sum(dim=-1).mean()
                loss_hard=torch.nn.functional.mse_loss(mu_s,mu_t)
                loss_pi=alpha*loss_soft+(1-alpha)*loss_hard
                # Value 損失 (保持不變)
                loss_vf = torch.nn.functional.mse_loss(values_s.squeeze(), values_t)
                
                # 總損失
                loss = loss_pi + vf_coef * loss_vf
                # -----------------------------------------------

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                run_pi_l += loss_pi.item() * obs.size(0)
                run_vf_l += loss_vf.item() * obs.size(0)
                n += obs.size(0)
            avg_pi_loss = run_pi_l / n
            avg_vf_loss = run_vf_l / n
            if avg_pi_loss > kl_target_max:
                temperature *= 1.15
                alpha = max(0.5, alpha - 0.05)
                print(f"🔥 KL高 {avg_pi_loss:.1f} → temp={temperature:.2f}, α={alpha:.2f}")
            elif avg_pi_loss < kl_target_min:
                temperature *= 0.9
                alpha = min(0.8, alpha + 0.05)
                print(f"🧊 KL低 {avg_pi_loss:.1f} → temp={temperature:.2f}, α={alpha:.2f}")
            else:
                print(f"✅ KL穩定 {avg_pi_loss:.1f}")

            print(f"Epoch {ep+1}: KL={avg_pi_loss:.3f}, Value_MSE={avg_vf_loss:.3f}")

        print("🎯 Offline distillation finished!\n")
                # 在損失名稱上做個區分
                
            
    policy.eval()
def _rebuild_policy_optimizer(policy: nn.Module):
    """
    跨 SB3 版本安全地重建 policy 的 optimizer。
    - 先取目前 requires_grad=True 的參數
    - 优先用 policy.setup_optimizer()（若存在）
    - 否則手動用 optimizer_class/optimizer_kwargs 建
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found when rebuilding optimizer.")

    # 一些 SB3 版本有 setup_optimizer()
    if hasattr(policy, "setup_optimizer") and callable(policy.setup_optimizer):
        # 注意：某些版本的 setup_optimizer 會對 self.parameters() 建 optimizer，
        # 會尊重 requires_grad 狀態；你已在外面設好 requires_grad 了。
        policy.setup_optimizer()
    else:
        # 後備方案：手動建立
        opt_cls = getattr(policy, "optimizer_class", torch.optim.Adam)
        opt_kwargs = getattr(policy, "optimizer_kwargs", {})
        policy.optimizer = opt_cls(trainable, **opt_kwargs)
def progressive_evolve(
    dad: PPO, mom: PPO, env, distill_pt,
    rounds: int = 1, finetune_steps: int = 150_00,debug_plan: Optional[Dict[int, Tuple[str, int]]] = None,
):
    child_policy = copy.deepcopy(dad.policy).to("cuda")  # 以父為骨架
    forbidden: Dict[str,Set[int]]={}
    for r in range(1, rounds+1):
        print(f"\n================  Round {r}  ================")
        round_debug_target = debug_plan.get(r) if debug_plan else None
        child_policy, info = crossover_one_boundary(
            child_policy, dad.policy, mom.policy,
            p_from_mother=0.5, device="cuda",forbidden=forbidden,debug_target=round_debug_target)
        print("插入 info:", info["adapter"])
        touched = info.get("touched", {})
        for sp, idx_set in touched.items():
            forbidden.setdefault(sp, set()).update(idx_set)
        # 用 child_policy 建一隻暫時的 PPO（共用超參）
        child = PPO(
            policy=dad.policy.__class__,
            env=env,
            verbose=0,
            learning_rate=3e-5,
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
        )
        child.policy = child_policy
        child.policy.to(child.device)
        _rebuild_policy_optimizer(child.policy)  # 讓 optimizer 知道哪些要訓練
        if info.get("adapter") is None:
            print("[offline] no new adapter this round -> skip offline distill")
        else:
            train_adapters_offline(
                child, distill_pt,adapter_info=info["adapter"],
                epochs=2, lr=1e-3, batch_size=4096,
                lambda_std=0.05, device="cuda", temperature=2.0  # 或 "cuda"
    )
        # 解凍全網 finetune
        unfreeze_all(child.policy)
        _rebuild_policy_optimizer(child.policy)
        child.set_env(env)
        child.learn(total_timesteps=finetune_steps, progress_bar=True)

        # 更新父引用 → 下一輪以最新子網再交配
        dad.policy = child.policy
        child_policy = child.policy

    return dad
def _unwrap_policy(x):
    return getattr(x, "policy", x)


def _safe_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu")

def print_and_dump_all_params(
    model: Union[nn.Module, object],
    who: str,
    out_dir: str = "./weight_dump",
    to_files: bool = True,
    print_full: bool = True,
    float_fmt: str = "{:.7g}",
):
    """
    完整列印 + 另存所有參數（weights/bias/ln 等）。
    - model: 可是 PPO 或 policy
    - who:   "dad" / "mom" / "child"（會用在檔名）
    - to_files: 另存到 out_dir/who/ 下（.pt + 各層 .txt）
    - print_full: True = 直接把整個 tensor 印到主控台
    """
    pol = _unwrap_policy(model)
    os.makedirs(os.path.join(out_dir, who), exist_ok=True)

    # 讓列印更完整（避免換行截斷）
    if print_full:
        torch.set_printoptions(sci_mode=False, linewidth=10_000, threshold=10_000_000)

    print(f"\n================= [{who}] 參數完整列印開始 =================")
    meta = {}

    for name, p in pol.named_parameters():
        t = _safe_cpu(p)
        shp = tuple(t.shape)
        meta[name] = {"shape": shp, "numel": t.numel()}

        header = f"\n--- {who} :: {name}  shape={shp}  numel={t.numel()} ---"
        print(header)
        if print_full:
            print(t)  # 直接完整印出
        else:
            # 只印摘要，避免刷屏
            print(f"min={t.min().item():.6g}  max={t.max().item():.6g}  mean={t.mean().item():.6g}  std={t.std().item():.6g}")

        if to_files:
            # 逐層文字檔（.txt）
            txt_path = os.path.join(out_dir, who, f"{name.replace('.', '_')}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                if print_full:
                    f.write(str(t) + "\n")
                else:
                    f.write(f"[summary] min={t.min().item():.9g} max={t.max().item():.9g} mean={t.mean().item():.9g} std={t.std().item():.9g}\n")

    # 存一份單檔 .pt（整包 state_dict）
    if to_files:
        state_pt = {k: _safe_cpu(v) for k, v in pol.state_dict().items()}
        torch.save(state_pt, os.path.join(out_dir, f"{who}_state_dict.pt"))
        with open(os.path.join(out_dir, f"{who}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n================= [{who}] 參數列印完成 =================\n"f"{'(已輸出至 ' + os.path.abspath(out_dir) + ')' if to_files else ''}")
@torch.no_grad()
def create_dual_channel_policy(dad_policy:nn.Module,mom_policy:nn.Module)->nn.Module:
    """
    child_policy = deepcopy(dad_policy) 之後，把 MLP 隱藏層拆半拼接（上半 dad、下半 mom）。

    action_net 不拆半：整層直接保留 dad 的原始權重（deepcopy 出來的預設值，這裡不覆寫）。
    原因：action_net 的輸出是「馬達扭矩」，4 個維度分別對應特定關節，不是「上半屬於 dad 身分、
    下半屬於 mom 身分」這種可切分的語意。且 action_net 被排除在所有 OT/TIES 交叉通道對齊之外
    （SKIP_LAYERS），如果像其他 Linear 層一樣硬拆半拼接，上半的 dad 權重仍會直接套用在完整的
    融合隱藏層（含未對齊的 mom 半邊）上，等於是套用一組從未校準過的權重去算最終動作，
    是最終動作輸出失真、子代一開始就摔倒的主因。因此整層固定用 dad 的權重，不參與拼接。
    """
    print("正在建構網路中.....")
    child_policy=copy.deepcopy(dad_policy)
    dad_modules=dict(dad_policy.named_modules())
    mom_modules=dict(mom_policy.named_modules())
    NO_SPLIT_LAYERS = {"action_net"}  # 整層固定用 dad 權重，不拆半、不拼接
    for name,child_module in child_policy.named_modules():
        if isinstance(child_module,nn.Linear):
            if any(s in name for s in NO_SPLIT_LAYERS):
                print(f"  - 層 '{name}' 不拆半，整層沿用 dad 權重。")
                continue
            dad_layer= dad_modules.get(name)
            mom_layer =mom_modules.get(name)
            if dad_layer is None or mom_layer is None:
                continue
            dad_w=dad_layer.weight.data
            mom_w=mom_layer.weight.data
            if dad_w.shape[0] % 2 != 0:
                print(f"  - 警告：層 '{name}' 的輸出維度 ({dad_w.shape[0]}) 為奇數，已跳過拼接。")
                continue
            out_half =dad_w.shape[0]//2
            assert dad_w.shape[0] % 2 == 0, f"Layer {name} out_features not divisible by 2"
            child_module.weight.data[:out_half,:]=dad_w[:out_half,:]
            child_module.weight.data[out_half:,:]=mom_w[out_half:,:]
            if child_module.bias is not None:
                dad_b=dad_layer.bias.data
                mom_b=mom_layer.bias.data
                child_module.bias.data[:out_half]=dad_b[:out_half]
                child_module.bias.data[out_half:]=mom_b[out_half:]

    print("網路建構完成")
    # verify_half_concat_mom_lower(child_policy,dad.policy,mom.policy)
    # print_and_dump_all_params(dad,   "dad",   to_files=True,  print_full=True)  # 爸爸
    # print_and_dump_all_params(mom,   "mom",   to_files=True,  print_full=True)  # 媽媽
    # print_and_dump_all_params(child_policy, "child", to_files=True,  print_full=True)
    return child_policy

@torch.no_grad()
def verify_half_concat_mom_lower(child: nn.Module, dad: nn.Module, mom: nn.Module, atol=1e-6) -> bool:
    """
    驗證「上半=爸爸上半，下半=媽媽下半」的拼接設計。
    """
    child = _unwrap_policy(child)
    dad = _unwrap_policy(dad)
    mom = _unwrap_policy(mom)
    ok = True
    dad_modules = dict(dad.named_modules())
    mom_modules = dict(mom.named_modules())

    for name, m in child.named_modules():
        if not isinstance(m, nn.Linear):
            continue

        d = dad_modules.get(name)
        mo = mom_modules.get(name)
        if d is None or mo is None:
            print(f"[SKIP] {name}: 找不到父母對應層")
            continue

        Wc, Wd, Wm = m.weight.data, d.weight.data, mo.weight.data
        if Wc.shape != Wd.shape or Wc.shape != Wm.shape or Wc.shape[0] % 2 != 0:
            print(f"[SKIP] {name}: 形狀不符 -> child:{Wc.shape}, dad:{Wd.shape}, mom:{Wm.shape}")
            continue

        oh = Wc.shape[0] // 2

        # 🧩 確認上半、下半分別來自父母正確區段
        diff_dad = (Wc[:oh, :] - Wd[:oh, :]).abs().max().item()     # 上半 = 爸上半
        diff_mom = (Wc[oh:, :] - Wm[oh:, :]).abs().max().item()     # 下半 = 媽下半
        cond_w = diff_dad <= atol and diff_mom <= atol

        # 偏置比對
        cond_b = True
        diff_b_dad = diff_b_mom = None
        if m.bias is not None:
            bd, bm, bc = d.bias.data, mo.bias.data, m.bias.data
            diff_b_dad = (bc[:oh] - bd[:oh]).abs().max().item()
            diff_b_mom = (bc[oh:] - bm[oh:]).abs().max().item()
            cond_b = diff_b_dad <= atol and diff_b_mom <= atol

        if cond_w and cond_b:
            print(f"[OK] {name:40s}  diff_dad={diff_dad:.3e}  diff_mom={diff_mom:.3e}")
        else:
            print(f"[X] {name:40s}  diff_dad={diff_dad:.3e}  diff_mom={diff_mom:.3e}")
            ok = False

    print("\n✅ 全部符合『上半=爸上半，下半=媽下半』" if ok else "\n⚠️ 有層未通過比對")
    return ok
@torch.no_grad()
def zero_initialize_crosstalk(policy: nn.Module):
    """
    將雙子網路中所有線性層的交叉通道權重歸零。
    """
    print("正在執行交叉通道歸零...")
    for name, module in policy.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            in_half = W.shape[1] // 2
            out_half = W.shape[0] // 2
            
            # 確保維度有效
            if in_half == 0 or out_half == 0: continue

            # # 左下角 (母 -> 父) 區塊
            # W[:out_half, in_half:].zero_()
            # # 右上角 (父 -> 母) 區塊
            # W[out_half:, :in_half].zero_()
            #--- 核心修改：從 .zero_() 改為 nn.init.normal_ ---
            ##選中右上角 (母 -> 父) 區塊
            cross_mf_block = W[:out_half, in_half:]
            nn.init.normal_(cross_mf_block, mean=0.0, std=0.001)

            # 選中左下角 (父 -> 母) 區塊
            cross_fm_block = W[out_half:, :in_half]
            nn.init.normal_(cross_fm_block, mean=0.0, std=0.001)
    print("歸零完成。")
    # print_and_dump_all_params(dad,   "dad",   to_files=True,  print_full=True)  # 爸爸
    # print_and_dump_all_params(mom,   "mom",   to_files=True,  print_full=True)  # 媽媽
    # print_and_dump_all_params(child_policy, "child", to_files=True,  print_full=True)

@torch.no_grad()
def _ties_block(dad_block: torch.Tensor, mom_block: torch.Tensor, k: float) -> torch.Tensor:
    """對兩個同形狀的區塊做 TIES 合併（flatten → trim → elect sign → disjoint merge → reshape）。"""
    shape = dad_block.shape
    d = dad_block.reshape(1, -1).float()
    m = mom_block.reshape(1, -1).float()
    M = torch.cat([d, m], dim=0)          # [2, D]
    base = M.mean(dim=0)                  # 用平均當 base
    tv = M - base.unsqueeze(0)            # task vectors [2, D]
    trimmed = _topk_trim(tv, k)           # trim
    sign = _resolve_sign(trimmed, "mass") # elect sign
    merged_tv = _disjoint_merge(trimmed, sign, "mean")  # disjoint merge
    merged = (base + merged_tv).reshape(shape)
    return merged

@torch.no_grad()
def ties_initialize_crosstalk(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    k: float = 0.2,
):
    """
    雙通道架構中，純通道區塊保持不動，
    交叉通道區塊改用 TIES 合併 dad 和 mom 的對應區塊。

    W [out, in]:
      左上 [:oh, :ih]  dad→dad  → 保留（來自 create_dual_channel_policy）
      右下 [oh:, ih:]  mom→mom  → 保留
      右上 [:oh, ih:]  mom→dad  → TIES(dad右上, mom右上)
      左下 [oh:, :ih]  dad→mom  → TIES(dad左下, mom左下)
    """
    print("[TIES crosstalk] 用 TIES 初始化交叉通道...")
    dad_modules = dict(dad_policy.named_modules())
    mom_modules = dict(mom_policy.named_modules())

    SKIP_LAYERS = {"action_net", "value_net"}
    for name, module in child_policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in SKIP_LAYERS):
            continue
        dad_m = dad_modules.get(name)
        mom_m = mom_modules.get(name)
        if dad_m is None or mom_m is None:
            continue

        W = module.weight.data
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        Wd = dad_m.weight.data
        Wm = mom_m.weight.data

        # 右上：mom→dad 交叉通道
        W[:oh, ih:] = _ties_block(Wd[:oh, ih:], Wm[:oh, ih:], k)
        # 左下：dad→mom 交叉通道
        W[oh:, :ih] = _ties_block(Wd[oh:, :ih], Wm[oh:, :ih], k)

    print("[TIES crosstalk] 完成。")


@torch.no_grad()
def ties_fuse_single_layer(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    stage_idx: int,
    k: float = 0.2,
) -> Optional[str]:
    """
    漸進式 TIES：只融合第 stage_idx 層（0-based）的交叉通道，其他層不動。
    TIES 各層獨立計算（不像 recursive OT 需要 T 逐層傳遞），所以可以任意層先做。
    回傳被融合的層名；stage_idx 超出層數回傳 None。
    """
    layer_names = _fusable_layer_names(child_policy)
    if stage_idx >= len(layer_names):
        return None

    name = layer_names[stage_idx]
    dad_modules   = dict(dad_policy.named_modules())
    mom_modules   = dict(mom_policy.named_modules())
    child_modules = dict(child_policy.named_modules())

    Wd = dad_modules[name].weight.data
    Wm = mom_modules[name].weight.data
    W  = child_modules[name].weight.data
    oh, ih = Wd.shape[0] // 2, Wd.shape[1] // 2

    W[:oh, ih:] = _ties_block(Wd[:oh, ih:], Wm[:oh, ih:], k)
    W[oh:, :ih] = _ties_block(Wd[oh:, :ih], Wm[oh:, :ih], k)
    return name


def progressive_ties_evolve(
    child_agent: "PPO",
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    k: float = 0.2,
    steps_per_stage: int = 20_000,
):
    """
    漸進式逐層 TIES 融合（退火版）：
      1. 尚未輪到的層維持 create_dual_channel_policy 拼接時的原始值（不歸零，理由同
         progressive_recursive_ot_evolve：dad/mom 自己的交叉通道對他們自己半邊仍有功能）
      2. stage k：用 TIES 把第 k 層的交叉通道覆寫成這一輪的合併值，
         幫這一層掛退火 hook，讓交叉通道貢獻在 steps_per_stage 步內從 0 線性升到 1，
         期間用 PPO 正常訓練全網（不凍結）
      3. alpha 升到 1 後移除該層的退火 hook
      4. 全部層融合完後，交還呼叫端做最終微調
    與 progressive_recursive_ot_evolve 共用同一套退火機制（attach/detach_crosstalk_annealing_hooks、
    CrosstalkAnnealCallback），差別只在每層的交叉通道值改用 TIES 合併而非 OT 翻譯。
    """
    policy = child_agent.policy
    layer_names = _fusable_layer_names(policy)
    n_layers = len(layer_names)
    print(f"[Progressive TIES] 共 {n_layers} 層將逐層融合（k={k}），每 stage 退火 {steps_per_stage} 步")

    _rebuild_policy_optimizer(policy)

    for stage in range(n_layers):
        fused_name = ties_fuse_single_layer(policy, dad_policy, mom_policy, stage, k=k)
        print(f"\n[Progressive TIES] stage {stage+1}/{n_layers}：已寫入層 {fused_name} 的 TIES 交叉通道，開始退火")

        if steps_per_stage > 0:
            alpha_boxes = attach_crosstalk_annealing_hooks(policy, [fused_name])
            anneal_cb = CrosstalkAnnealCallback(alpha_boxes, total_anneal_steps=steps_per_stage)
            child_agent.learn(total_timesteps=steps_per_stage, callback=[anneal_cb], progress_bar=True)
            detach_crosstalk_annealing_hooks(policy, [fused_name])

    print("[Progressive TIES] 全部層融合完成。")


def _ot_block(dad_block: torch.Tensor, mom_block: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    OT Fusion for one crosstalk block (weight-based alignment, exact EMD).
    Rows = output neurons.
    Steps:
      1. Cost matrix C[i,j] = ||dad_row_i - mom_row_j||^2
      2. T = ot.emd(uniform_mu, uniform_nu, C)   (exact optimal transport)
      3. Normalize T columns: T = T * diag(1 / (T.t() @ ones))
      4. aligned_dad = T.t() @ dad  (soft alignment toward mom's ordering)
      5. fused = 0.5 * (aligned_dad + mom)
    """
    import ot as pot
    shape = dad_block.shape
    n = shape[0]
    d = dad_block.float().reshape(n, -1)   # (n, d_in)
    m = mom_block.float().reshape(n, -1)

    # 1. Cost matrix (pairwise L2^2 between rows)
    diff = d.unsqueeze(1) - m.unsqueeze(0)   # (n, n, d_in)
    C = (diff ** 2).sum(dim=-1).cpu().numpy()

    # 2. Exact OT with uniform marginals
    mu = np.ones(n, dtype=np.float64) / n
    nu = np.ones(n, dtype=np.float64) / n
    T = pot.emd(mu, nu, C)                   # (n, n) numpy
    T_var = torch.from_numpy(T).float().to(d.device)

    # 3. Normalize columns so each column sums to 1 (proper_marginals correction)
    marginals_beta = T_var.t() @ torch.ones(n, device=d.device)
    T_var = T_var * (1.0 / (marginals_beta + eps))

    # 4. Soft-align dad toward mom's neuron ordering
    aligned_dad = T_var.t() @ d              # (n, d_in)

    # 5. Average
    fused = 0.5 * (aligned_dad + m)
    return fused.reshape(shape).to(dad_block.dtype)


@torch.no_grad()
def ot_initialize_crosstalk(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    ot_frac: float = 1.0,
):
    """
    雙通道架構中，交叉通道區塊用 OT Fusion 初始化。
    純通道區塊（左上/右下）保持不動。

    W [out, in]:
      左上 [:oh, :ih]  dad→dad  → 保留
      右下 [oh:, ih:]  mom→mom  → 保留
      右上 [:oh, ih:]  mom→dad  → OT_block(dad右上, mom右上)
      左下 [oh:, :ih]  dad→mom  → OT_block(dad左下, mom左下)

    ot_frac：只把交叉區塊「前 ot_frac 比例的列」覆寫成 OT 值，其餘列先歸零（小噪音）
    再維持不動，而不是留著 dad/mom 未經對齊的原始值。ot_frac=1.0 為原本行為（整塊覆寫）。
    """
    print(f"[OT crosstalk] 用 OT Fusion 初始化交叉通道（ot_frac={ot_frac}）...")
    dad_modules = dict(dad_policy.named_modules())
    mom_modules = dict(mom_policy.named_modules())

    SKIP_LAYERS = {"action_net", "value_net"}
    for name, module in child_policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in SKIP_LAYERS):
            continue
        dad_m = dad_modules.get(name)
        mom_m = mom_modules.get(name)
        if dad_m is None or mom_m is None:
            continue

        W = module.weight.data
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        Wd = dad_m.weight.data
        Wm = mom_m.weight.data

        k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
        if k < oh:
            # 沒被 OT 覆寫的列先歸零（小噪音），不留 dad/mom 未對齊的原始值
            nn.init.normal_(W[:oh, ih:], mean=0.0, std=0.001)
            nn.init.normal_(W[oh:, :ih], mean=0.0, std=0.001)
            print(f"  [partial OT] 層 {name}：只覆寫交叉通道前 {k}/{oh} 列，其餘歸零")

        if k > 0:
            # 右上：mom→dad 交叉通道（前 k 列）
            W[:k, ih:] = _ot_block(Wd[:k, ih:], Wm[:k, ih:])
            # 左下：dad→mom 交叉通道（前 k 列）
            W[oh:oh + k, :ih] = _ot_block(Wd[oh:oh + k, :ih], Wm[oh:oh + k, :ih])

    print("[OT crosstalk] 完成。")


def _compute_layer_transport(dad_W: torch.Tensor, mom_W: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    計算 dad 和 mom 純通道權重之間的 OT 傳輸矩陣 T。
    T[i,j] = dad 第 i 顆輸出神經元對應 mom 第 j 顆的權重。
    Column-normalize 使每列和=1（軟置換）。
    返回 (n, n) float tensor，在 dad_W.device 上。
    """
    import ot as pot
    n = dad_W.shape[0]
    d = dad_W.float().reshape(n, -1)
    m = mom_W.float().reshape(n, -1)
    diff = d.unsqueeze(1) - m.unsqueeze(0)
    C = (diff ** 2).sum(dim=-1).cpu().numpy()
    mu = np.ones(n, dtype=np.float64) / n
    nu = np.ones(n, dtype=np.float64) / n
    T = pot.emd(mu, nu, C)
    T_var = torch.from_numpy(T).float().to(dad_W.device)
    col_sums = T_var.sum(dim=0).clamp(min=eps)
    T_var = T_var / col_sums.unsqueeze(0)
    return T_var  # (n, n)


@torch.no_grad()
def _sample_observations(env, n_samples: int = 512) -> torch.Tensor:
    """
    從 env 隨機互動蒐集一批具代表性的 observation，供 activation-based OT 對齊使用。
    用隨機動作在多個 episode 間取樣（而非只用初始狀態），相容 gym / gymnasium 兩種 API。
    """
    obs_list = []
    obs = env.reset()
    if isinstance(obs, tuple):  # gymnasium: (obs, info)
        obs = obs[0]
    while len(obs_list) < n_samples:
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:  # gymnasium: obs, reward, terminated, truncated, info
            obs, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:  # 舊版 gym: obs, reward, done, info
            obs, _, done, _ = step_out
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    obs_arr = np.stack(obs_list[:n_samples], axis=0)
    return torch.from_numpy(obs_arr).float()


@torch.no_grad()
def _collect_pure_activations(
    policy: nn.Module, obs_batch: torch.Tensor, layer_names: List[str]
) -> Dict[str, torch.Tensor]:
    """
    對 policy 做一次真實前向傳播，擷取每個指定 Linear 層的 pre-activation 輸出（Tanh 之前）。
    回傳 {層名: (m, out_features) tensor}，m = obs_batch 筆數。
    """
    device = next(policy.parameters()).device
    obs_dev = obs_batch.to(device)
    acts: Dict[str, torch.Tensor] = {}
    hooks = []
    named = dict(policy.named_modules())

    def make_hook(name):
        def hook(module, inp, out):
            acts[name] = out.detach().float().cpu()
        return hook

    for name in layer_names:
        m = named.get(name)
        if m is not None:
            hooks.append(m.register_forward_hook(make_hook(name)))

    features = policy.extract_features(obs_dev)
    if policy.share_features_extractor:
        policy.mlp_extractor(features)
    else:
        policy.mlp_extractor(features, None)

    for h in hooks:
        h.remove()
    return acts


def _compute_layer_transport_from_acts(
    dad_acts: torch.Tensor, mom_acts: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Activation-based ground metric：用真實觀測下的神經元反應計算 OT 傳輸矩陣，
    取代權重向量比較（通常能找到更貼近「功能相似」而非「數值相似」的對齊，殘留誤差更小）。
    dad_acts, mom_acts: (m, n)，m 筆樣本、n 顆神經元。回傳 (n, n) 傳輸矩陣。
    """
    import ot as pot
    n = dad_acts.shape[1]
    d = dad_acts.t().float()   # (n, m)
    m_ = mom_acts.t().float()  # (n, m)
    diff = d.unsqueeze(1) - m_.unsqueeze(0)
    C = (diff ** 2).sum(dim=-1).cpu().numpy()
    mu = np.ones(n, dtype=np.float64) / n
    nu = np.ones(n, dtype=np.float64) / n
    T = pot.emd(mu, nu, C)
    T_var = torch.from_numpy(T).float()
    col_sums = T_var.sum(dim=0).clamp(min=eps)
    T_var = T_var / col_sums.unsqueeze(0)
    return T_var  # (n, n)


@torch.no_grad()
def recursive_ot_initialize_crosstalk(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    env=None,
    obs_batch: Optional[torch.Tensor] = None,
    n_obs_samples: int = 512,
    ot_frac: float = 1.0,
):
    """
    Recursive OT 交叉通道初始化（純通道完全不動，交叉通道對齊後取平均）：

    ★ T 的計算依據是「交叉通道本身」，只維護單一一條 T 鏈：
      比對的兩個區塊選「鏡像位置」：dad 自己的 X_md 交叉區塊 Wd[:oh,ih:]（右上）
      vs mom 自己的 X_dm 交叉區塊 Wm[oh:,:ih]（左下），用同一個 T 處理 X_md 方向、
      T 的轉置處理 X_dm 方向（跟原本純通道版本用對角區塊算 T 是同一種對稱邏輯，只是換成反對角）。

    ★ 跟論文（wasserstein_ensemble.py）一致：上一層的 T 不只用來決定「這一層寫入什麼值」，
      也真正參與「這一層新 T 怎麼算」——先用上一層的 T 把 dad 的交叉區塊 Wd_cross 的輸入端
      翻譯過（Wd_cross_aligned = Wd_cross @ T_prev），才拿翻譯後的結果去跟 mom_cross 算這一層
      的新 T，而不是每層各自獨立比較原始交叉權重。第一層沒有上一層可用，T_prev=None，
      不翻譯直接比較。

    - 第一層：T_0 = I（共用觀測空間），交叉通道 = 純通道權重（精確，無近似，不需要平均）。
      這一層結束後仍要算出 T 交給下一層，此時 T_prev 還是 None，不做輸入端翻譯，直接比較
      Wd_cross、Wm_cross。
    - 第 l 層（l>=2）：
        X_md = 0.5 * (Wd_pure @ T_{l-1} + Wm[:oh, ih:])
        X_dm = 0.5 * (Wm_pure @ T_{l-1}.T + Wd[oh:, :ih])
      純通道維持 create_dual_channel_policy 拼接時的父/母原始權重，完全不寫入。

    對齊方式（ground metric）：全部層一律 weight-based，直接比較（翻譯過的）交叉權重向量的
    數值差距，不使用真實觀測驅動的 activation-based 對齊（不需要 env/obs_batch，也不依賴
    抽樣品質，每次結果都是確定性的）。

    注意：即使有平均，振幅仍可能因疊加而偏大，這裡不在初始化階段處理，改由
    attach_crosstalk_annealing_hooks() 在訓練期間把交叉通道貢獻從 0 退火升到 1。
    """
    print("[Recursive OT] 逐層 OT 傳輸矩陣初始化交叉通道（依據=交叉通道，weight-based，T 鏈參與 ground metric，純通道不動）...")
    import ot as pot  # noqa: ensure imported

    SKIP_LAYERS = {"action_net", "value_net"}
    dad_modules = dict(dad_policy.named_modules())
    mom_modules = dict(mom_policy.named_modules())

    layer_names = [
        name for name, module in child_policy.named_modules()
        if isinstance(module, nn.Linear) and not any(s in name for s in SKIP_LAYERS)
    ]

    T_prev = None     # 單一 T 鏈（None = 第一層，identity）

    for name in layer_names:
        child_m = dict(child_policy.named_modules()).get(name)
        dad_m = dad_modules.get(name)
        mom_m = mom_modules.get(name)
        if child_m is None or dad_m is None or mom_m is None:
            continue

        W = child_m.weight.data
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        Wd = dad_m.weight.data
        Wm = mom_m.weight.data

        # 純通道區塊（僅用於「寫入交叉通道」的翻譯計算，不寫回，純通道保持不動）
        Wd_pure = Wd[:oh, :ih]  # (oh, ih)
        Wm_pure = Wm[oh:, ih:]  # (oh, ih)

        # 交叉通道區塊（鏡像位置：dad 的 X_md vs mom 的 X_dm，T 的計算依據）
        Wd_cross = Wd[:oh, ih:]   # dad 自己的 X_md 交叉區塊（右上）
        Wm_cross = Wm[oh:, :ih]   # mom 自己的 X_dm 交叉區塊（左下）

        # ot_frac：只把前 k 列（輸出神經元）寫成 OT 對齊值，其餘列維持
        # create_dual_channel_policy 拼接時留下的 dad/mom 原始值，不歸零、不訓練。
        k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
        if k < oh:
            print(f"  [partial OT] 層 {name}：只覆寫交叉通道前 {k}/{oh} 列，其餘維持原始值")

        if T_prev is None:
            # 第一層：T_0 = I，交叉通道 = 純通道權重（精確，無近似，不需要平均）
            if k > 0:
                W[:k, ih:] = Wd_pure[:k].clone()
                W[oh:oh + k, :ih] = Wm_pure[:k].clone()
            Wd_cross_aligned = Wd_cross.float()  # 沒有上一層 T，不翻譯
        else:
            # 深層：交叉通道 = 0.5 * (OT 翻譯值 + 對方自己那格的交叉通道)
            aligned_dad = (Wd_pure.float() @ T_prev.to(Wd_pure.device))
            aligned_mom = (Wm_pure.float() @ T_prev.t().to(Wm_pure.device))
            if k > 0:
                W[:k, ih:] = (0.5 * (aligned_dad[:k] + Wm[:k, ih:].float())).to(W.dtype)
                W[oh:oh + k, :ih] = (0.5 * (aligned_mom[:k] + Wd[oh:oh + k, :ih].float())).to(W.dtype)

            # ★ 跟論文一致：先用上一層的 T 翻譯 dad 交叉區塊的輸入端，再拿去算這一層的新 T
            # （T 鏈的計算不受 ot_frac 影響，永遠用完整的交叉區塊）
            Wd_cross_aligned = (Wd_cross.float() @ T_prev.to(Wd_cross.device))

        # 計算本層的 T（用翻譯過的 Wd_cross_aligned，而非原始 Wd_cross），傳遞給下一層
        # 全部層一律 weight-based：直接比較（翻譯過的）交叉權重向量
        T_prev = _compute_layer_transport(Wd_cross_aligned, Wm_cross)

    print("[Recursive OT] 完成。")


@torch.no_grad()
def iterative_ot_initialize_crosstalk(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    n_iters: int = 5,
    ot_frac: float = 1.0,
):
    """
    迭代 OT 交叉通道初始化：
      - 第 1 次迭代：與 recursive_ot 相同，用父母純通道算 T，填入交叉通道。
      - 第 k 次迭代：用上一次的交叉通道結果（X_md, X_dm）重新算 T，再做一次 W@T。
      - 多次迭代讓 T 越來越精準，交叉通道逐漸收斂對齊。
    純通道（左上/右下）全程不動。

    ot_frac：疊代過程照常在完整的交叉通道區塊上進行（T 的計算需要完整列空間），
    但最終寫回網路時，只保留前 ot_frac 比例的列是疊代收斂值，其餘列改成歸零
    （小噪音），不是疊代結果。ot_frac=1.0 為原本行為（整塊都是疊代結果）。
    """
    print(f"[Iterative OT] 開始迭代對齊，共 {n_iters} 次...")
    import ot as pot

    SKIP_LAYERS = {"action_net", "value_net"}
    dad_modules = dict(dad_policy.named_modules())
    mom_modules = dict(mom_policy.named_modules())

    layer_names = [
        name for name, module in child_policy.named_modules()
        if isinstance(module, nn.Linear) and not any(s in name for s in SKIP_LAYERS)
    ]

    for it in range(n_iters):
        child_modules = dict(child_policy.named_modules())
        T_prev = None

        for name in layer_names:
            child_m = child_modules.get(name)
            dad_m   = dad_modules.get(name)
            mom_m   = mom_modules.get(name)
            if child_m is None or dad_m is None or mom_m is None:
                continue

            W  = child_m.weight.data
            oh, ih = W.shape[0] // 2, W.shape[1] // 2
            if oh == 0 or ih == 0:
                continue

            Wd = dad_m.weight.data
            Wm = mom_m.weight.data

            # 純通道（用於計算 T）
            Wd_pure = Wd[:oh, :ih]
            Wm_pure = Wm[oh:, ih:]

            if it == 0:
                # 第一次迭代：與 recursive_ot 相同
                if T_prev is None:
                    W[:oh, ih:] = Wd_pure.clone()
                    W[oh:, :ih] = Wm_pure.clone()
                else:
                    W[:oh, ih:] = (Wd_pure.float() @ T_prev.to(Wd_pure.device)).to(W.dtype)
                    W[oh:, :ih] = (Wm_pure.float() @ T_prev.t().to(Wm_pure.device)).to(W.dtype)
                T_prev = _compute_layer_transport(Wd_pure, Wm_pure)
            else:
                # 後續迭代：讓 X_md 對齊 W_dad_pure 的列空間，X_dm 對齊 W_mom_pure 的列空間
                # 每輪讓交叉通道慢慢靠近父母純通道，T 作用在 input 維度 → shape (ih, ih)
                X_md_prev = W[:oh, ih:].clone()  # (oh, ih)
                X_dm_prev = W[oh:, :ih].clone()  # (oh, ih)
                dev = X_md_prev.device

                # T_md: 讓 X_md 的列空間對齊 W_dad_pure 的列空間
                T_md = _compute_layer_transport(
                    X_md_prev.t().contiguous(),   # (ih, oh)
                    Wd_pure.t().contiguous(),      # (ih, oh)
                )  # (ih, ih)
                # T_dm: 讓 X_dm 的列空間對齊 W_mom_pure 的列空間
                T_dm = _compute_layer_transport(
                    X_dm_prev.t().contiguous(),   # (ih, oh)
                    Wm_pure.t().contiguous(),      # (ih, oh)
                )  # (ih, ih)

                W[:oh, ih:] = (X_md_prev.float() @ T_md.to(dev)).to(W.dtype)
                W[oh:, :ih] = (X_dm_prev.float() @ T_dm.to(dev)).to(W.dtype)
                # 傳給下一層的 T 用 T_md（代表 dad 側的對齊）
                T_prev = T_md

        # 計算本次迭代的交叉通道變化量（收斂指標）
        xmd = torch.cat([dict(child_policy.named_modules())[n].weight.data[:dict(child_policy.named_modules())[n].weight.data.shape[0]//2, dict(child_policy.named_modules())[n].weight.data.shape[1]//2:].flatten()
                         for n in layer_names if dict(child_policy.named_modules()).get(n) is not None
                         and dict(child_policy.named_modules())[n].weight.data.shape[0] > 1])
        print(f"  iter {it+1}/{n_iters}  X_md norm={xmd.norm():.4f}")

    # 迭代完成後：與父母純通道做 0.5 平均（防止多次 @T 造成數值收縮，同時保留對齊結果）
    print("[Iterative OT] 迭代完成，對交叉通道做最終平均...")
    dad_modules_final = dict(dad_policy.named_modules())
    mom_modules_final = dict(mom_policy.named_modules())
    for name in layer_names:
        child_m = dict(child_policy.named_modules()).get(name)
        dad_m   = dad_modules_final.get(name)
        mom_m   = mom_modules_final.get(name)
        if child_m is None or dad_m is None or mom_m is None:
            continue
        W  = child_m.weight.data
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue
        Wd_pure = dad_m.weight.data[:oh, :ih]
        Wm_pure = mom_m.weight.data[oh:, ih:]
        # X_md_final = 0.5 * (迭代收斂值 + dad 純通道原始值)
        X_md_final = 0.5 * (W[:oh, ih:].float() + Wd_pure.float())
        # X_dm_final = 0.5 * (迭代收斂值 + mom 純通道原始值)
        X_dm_final = 0.5 * (W[oh:, :ih].float() + Wm_pure.float())

        k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
        if k < oh:
            print(f"  [partial OT] 層 {name}：只保留前 {k}/{oh} 列為疊代結果，其餘歸零")
            nn.init.normal_(W[:oh, ih:], mean=0.0, std=0.001)
            nn.init.normal_(W[oh:, :ih], mean=0.0, std=0.001)
        if k > 0:
            W[:k, ih:] = X_md_final[:k].to(W.dtype)
            W[oh:oh + k, :ih] = X_dm_final[:k].to(W.dtype)
    print("[Iterative OT] 完成。")


def _sample_obs_batch(env, n: int = 256) -> Optional[torch.Tensor]:
    """env 隨機跑 n 步收集觀測，回傳 (n, obs_dim) tensor；env=None 時回傳 None。"""
    if env is None:
        return None
    obs_list = []
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        for _ in range(n):
            obs_list.append(obs if isinstance(obs, np.ndarray) else np.array(obs))
            action = env.action_space.sample()
            result = env.step(action)
            obs, done = result[0], result[2]
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
    except Exception as e:
        print(f"[_sample_obs_batch] 收集失敗：{e}，改用 weight-based T")
        traceback.print_exc()
        return None
    return torch.from_numpy(np.stack(obs_list)).float()


def mutual_ot_initialize_crosstalk(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    n_iters: int = 20,
    tol: float = 1e-3,
    obs_batch: Optional[torch.Tensor] = None,
    ot_frac: float = 1.0,
):
    """
    互相 OT 融合：把 X_md 和 X_dm 當成兩個不同的模型，
    反覆套用論文 OT Fusion 的做法直到兩者收斂對齊。

    每輪（單向，見下方主迴圈註解）：
      T = OT(X_md, X_dm)
      X_dm ← 0.5 * (T@X_dm + X_md)   ← X_dm 吸收對齊後的結果，X_md 錨點全程不動

    當所有層的 |X_dm - X_md| 總 norm < tol 時提早停止。
    純通道（左上/右下）全程不動。

    ot_frac：seed 階段與主疊代迴圈都只處理交叉通道「前 ot_frac 比例的列」，其餘列
    完全不寫入，維持 create_dual_channel_policy 拼接時留下的 dad/mom 原始值不變
    （不歸零）。ot_frac=1.0 為原本行為（整塊都參與 OT）。
    """
    print(f"[Mutual OT] 開始兩通道互相 OT 融合，最多 {n_iters} 輪，收斂門檻={tol}，ot_frac={ot_frac}...")

    SKIP_LAYERS = {"action_net", "value_net"}
    dad_modules = dict(dad_policy.named_modules())
    mom_modules = dict(mom_policy.named_modules())

    layer_names = [
        name for name, module in child_policy.named_modules()
        if isinstance(module, nn.Linear) and not any(s in name for s in SKIP_LAYERS)
    ]

    # 初始化：與 recursive_ot 相同（第一層 T=I，後續逐層傳 T）
    T_prev = None
    child_modules = dict(child_policy.named_modules())
    for name in layer_names:
        child_m = child_modules.get(name)
        dad_m   = dad_modules.get(name)
        mom_m   = mom_modules.get(name)
        if child_m is None or dad_m is None or mom_m is None:
            continue
        W  = child_m.weight.data
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue
        Wd_pure = dad_m.weight.data[:oh, :ih]
        Wm_pure = mom_m.weight.data[oh:, ih:]
        k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
        if k > 0:
            if T_prev is None:
                W[:k, ih:] = Wd_pure[:k].clone()
                W[oh:oh + k, :ih] = Wm_pure[:k].clone()
            else:
                W[:k, ih:] = (Wd_pure[:k].float() @ T_prev.to(Wd_pure.device)).to(W.dtype)
                W[oh:oh + k, :ih] = (Wm_pure[:k].float() @ T_prev.t().to(Wm_pure.device)).to(W.dtype)
        # k < oh 的其餘列完全不寫入，維持 create_dual_channel_policy 留下的原始值
        T_prev = _compute_layer_transport(Wd_pure, Wm_pure)

    use_acts = obs_batch is not None
    if use_acts:
        print(f"[Mutual OT] 使用 activation-based T（{obs_batch.shape[0]} 筆觀測）")
    else:
        print("[Mutual OT] 使用 weight-based T（無觀測資料）")

    # 迭代：把 X_md、X_dm 當成兩個「模型」，改成 otfusion 官方的單向做法——
    # X_md 當作 model1，全程完全不動；每輪只把 X_dm（model0）用當輪的 T 翻譯到
    # X_md 的座標系，再跟「完全沒被動過」的 X_md 取平均，寫回 X_dm。
    # 每一輪內部逐層處理時，仿照 otfusion 的做法用 T_prev 鏈式傳遞：
    # 上一層算出的 T，先拿來修正這一層的 X_dm，再算這一層自己的 T。
    for it in range(n_iters):
        child_modules = dict(child_policy.named_modules())

        # 若有 obs_batch，每輪重新收集 child 當前 activation（T 因此每輪不同）
        if use_acts:
            with torch.no_grad():
                child_acts = _collect_pure_activations(child_policy, obs_batch, layer_names)

        total_gap_sq = 0.0
        T_prev = None  # 這一輪逐層鏈式傳遞用：上一層算出的 T，先拿來修正這一層的 X_dm
        for name in layer_names:
            child_m = child_modules.get(name)
            if child_m is None:
                continue
            W  = child_m.weight.data
            oh, ih = W.shape[0] // 2, W.shape[1] // 2
            if oh == 0 or ih == 0:
                continue

            X_md = W[:oh, ih:].clone().float()  # model1：錨點，全程不動
            X_dm = W[oh:, :ih].clone().float()  # model0：每輪被翻譯、更新
            dev  = W.device

            if T_prev is not None:
                # X_dm 的欄（ih）＝上一層的 dad 輸出神經元，跟 T_prev 的索引意義一致，
                # 所以要「右乘」作用在欄上，而不是左乘作用在列（這一層自己的 mom 輸出神經元）上。
                X_dm = X_dm @ T_prev.to(dev).t()

            if use_acts and name in child_acts:
                acts = child_acts[name]          # (N, 2*oh)
                acts_upper = acts[:, :oh]        # (N, oh) ← dad 神經元 activation
                acts_lower = acts[:, oh:]        # (N, oh) ← mom 神經元 activation
                T = _compute_layer_transport_from_acts(acts_upper, acts_lower)
            else:
                T = _compute_layer_transport(X_md, X_dm)  # weight-based fallback

            T = T.to(dev)
            t_dm = T @ X_dm                 # 把（校正過的）X_dm 翻譯到 X_md 的座標系
            X_dm_new = 0.5 * (t_dm + X_md)  # 跟完全沒動過的 X_md（model1）取平均

            k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
            if k > 0:
                total_gap_sq += (X_dm_new[:k] - X_md[:k]).pow(2).sum().item()  # X_dm 離目標 X_md 還有多遠
                W[oh:oh + k, :ih] = X_dm_new[:k].to(W.dtype)
            # k < oh 的其餘列完全不寫入，維持 create_dual_channel_policy 留下的原始值
            # X_md（W[:oh, ih:]）維持不動

            T_prev = T  # 把這層的 T 留給下一層鏈式使用

        gap = total_gap_sq ** 0.5
        print(f"  iter {it+1}/{n_iters}  |X_dm - X_md|（是否逐漸對齊）={gap:.4f}")
        if gap < tol:
            print(f"  [收斂] |X_dm - X_md|={gap:.6f} < tol={tol}，已對齊，提早停止（第 {it+1} 輪）")
            break

    print("[Mutual OT] 完成。")


@torch.no_grad()
def reconstruct_ot_initialize_crosstalk(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    env=None,
    obs_batch: Optional[torch.Tensor] = None,
    n_obs_samples: int = 512,
):
    """
    「重建被截肢的另一半」交叉通道初始化：
      拼接（create_dual_channel_policy）發生的瞬間，dad 上半神經元原本依賴的「dad 自己的下半
      輸入」被替換成「mom 的下半」，等於截肢；dad 自己的右上交叉區塊 Wd[:oh, ih:] 原本就是
      「處理下半輸入」的權重，只是現在下半換成了 mom，訊號對不上。

      修法：用 OT 把「mom 下半」翻譯成「dad 自己原本下半該有的樣子」，讓 dad 自己的交叉權重
      繼續work：
        X_md = Wd[:oh, ih:] @ T_bb   （T_bb：dad自己下半 vs mom自己下半，兩邊都是「母系血統」半層）
        X_dm = Wm[oh:, :ih] @ T_tt   （T_tt：dad自己上半 vs mom自己上半，兩邊都是「父系血統」半層）
      不做平均——這是重建单一血統的訊號，不是融合兩個血統。

      第一層特殊：輸入是共用觀測值，沒有「上半/下半」語意，create_dual_channel_policy 拼接時
      放的 Wd[:oh, ih:]（dad 原生右上區塊）本來就是精確解，完全不覆寫。

      T_bb / T_tt 每層獨立算（不像 recursive_ot 需要逐層傳遞），因為直接拿 dad_policy /
      mom_policy 各自完整原生前向傳播的真實激活值來比對，不依賴前面層的翻譯結果。
    """
    print("[Reconstruct OT] 重建被截肢半層的交叉通道...")

    SKIP_LAYERS = {"action_net", "value_net"}
    dad_modules = dict(dad_policy.named_modules())
    mom_modules = dict(mom_policy.named_modules())

    layer_names = [
        name for name, module in child_policy.named_modules()
        if isinstance(module, nn.Linear) and not any(s in name for s in SKIP_LAYERS)
    ]

    use_acts = False
    dad_acts_by_layer = mom_acts_by_layer = None
    if obs_batch is None and env is not None:
        obs_batch = _sample_observations(env, n_samples=n_obs_samples)
    if obs_batch is not None:
        dad_acts_by_layer = _collect_pure_activations(dad_policy, obs_batch, layer_names)
        mom_acts_by_layer = _collect_pure_activations(mom_policy, obs_batch, layer_names)
        use_acts = True
        print(f"[Reconstruct OT] 使用 activation-based 對齊（{obs_batch.shape[0]} 筆真實觀測）")
    else:
        print("[Reconstruct OT] 未提供 env/obs_batch，退回 weight-based 對齊")

    for i, name in enumerate(layer_names):
        child_m = dict(child_policy.named_modules()).get(name)
        dad_m = dad_modules.get(name)
        mom_m = mom_modules.get(name)
        if child_m is None or dad_m is None or mom_m is None:
            continue

        W = child_m.weight.data
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        Wd = dad_m.weight.data
        Wm = mom_m.weight.data

        if i == 0:
            # 第一層：輸入是共用觀測值，create_dual_channel_policy 放的原始值已是精確解，不動
            print(f"  [{name}] 第一層，維持原始拼接值（精確解，不需翻譯）")
            continue

        # 深層：Wd 自己的下半(母系血統) vs Wm 自己的下半(母系血統) → T_bb
        #       Wd 自己的上半(父系血統) vs Wm 自己的上半(父系血統) → T_tt
        if use_acts:
            dad_bottom = dad_acts_by_layer[name][:, oh:]   # dad 自己下半的真實反應
            mom_bottom = mom_acts_by_layer[name][:, oh:]   # mom 自己下半的真實反應
            dad_top    = dad_acts_by_layer[name][:, :oh]   # dad 自己上半的真實反應
            mom_top    = mom_acts_by_layer[name][:, :oh]   # mom 自己上半的真實反應
            T_bb = _compute_layer_transport_from_acts(dad_bottom, mom_bottom).to(W.device)
            T_tt = _compute_layer_transport_from_acts(dad_top, mom_top).to(W.device)
        else:
            T_bb = _compute_layer_transport(Wd[oh:, ih:], Wm[oh:, ih:])
            T_tt = _compute_layer_transport(Wd[:oh, :ih], Wm[:oh, :ih])

        Wd_cross = Wd[:oh, ih:].float()   # dad 自己原生的右上交叉區塊
        Wm_cross = Wm[oh:, :ih].float()   # mom 自己原生的左下交叉區塊

        W[:oh, ih:] = (Wd_cross @ T_bb.to(Wd_cross.device)).to(W.dtype)
        W[oh:, :ih] = (Wm_cross @ T_tt.to(Wm_cross.device)).to(W.dtype)

    print("[Reconstruct OT] 完成。")


def _fusable_layer_names(policy: nn.Module) -> List[str]:
    """
    回傳可二分融合的 Linear 層名。
    只跳過 action_net——它的輸出是固定語意的物理量（各關節扭矩），神經元不可互換，
    OT/交叉通道對齊在這裡沒有意義，且已驗證過拆分會導致 child 一開始就崩潰。
    value_net 的隱藏層（value_net.0 / .2）*不*跳過：漸進式融合每個 stage 都會用真實
    PPO 在環境裡訓練，訓練品質依賴 value function 的 advantage 估計，所以這裡的交叉
    通道也需要對齊。value_net 最終輸出層（1 維）會被下面的形狀檢查自動排除，不需要
    額外過濾。
    """
    SKIP_LAYERS = {"action_net"}
    names = []
    for name, m in policy.named_modules():
        if not isinstance(m, nn.Linear) or any(s in name for s in SKIP_LAYERS):
            continue
        W = m.weight.data
        if W.shape[0] // 2 > 0 and W.shape[1] // 2 > 0:
            names.append(name)
    return names


@torch.no_grad()
def recursive_ot_fuse_single_layer(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    stage_idx: int,
    ot_frac: float = 1.0,
    align_first_layer: bool = False,
    average_with_parent: bool = True,
    alpha: Optional[float] = None,
) -> Optional[str]:
    """
    漸進式 Recursive OT：只融合第 stage_idx 層（0-based）的交叉通道，其他層不動。
    T 由 dad/mom 原始純通道權重逐層遞迴計算——父母權重固定，
    因此不論 child 在前面 stage 被微調成什麼樣，各 stage 算出的 T 序列一致。
    純通道完全不寫入；交叉通道 = 0.5*(OT 翻譯值 + 對方自己那格的交叉通道)，
    與 ot_initialize_crosstalk 的 _ot_block 平均風格一致（對稱版：X_md 配 mom 自己那格，
    X_dm 配 dad 自己那格）。振幅問題交由退火 alpha 處理。

    ot_frac：只把交叉區塊「前 ot_frac 比例的列（輸出神經元）」覆寫成 OT 值，
    其餘列維持原值（例如蒸餾基底）——用來降低 OT 的「劑量」，
    ot_frac=1.0 為原本行為（整個交叉區塊都覆寫）。

    align_first_layer：預設 False，維持原本行為——第一層（或剛跨分支後的第一層）沒有
    上一層可用的 T_prev，直接精確複製 dad/mom 純通道（T=I，不算 OT）。
    設 True 時改成真的算一次 OT：拿 Wd_pure（dad 自己 oh 個隱藏神經元的權重）跟 Wm_pure
    （mom 自己 oh 個隱藏神經元的權重）比對「誰的功能最像誰」，算出 (oh, oh) 的配對 T，
    再用 T 把 Wd_pure 的神經元順序翻譯成 mom 的排列（T @ Wd_pure）、Wm_pure 翻譯成 dad
    的排列（T.T @ Wm_pure），取代原本的精確複製直接寫入——注意這裡的 T 是對「輸出（神經元）
    維度」配對，跟深層分支用來翻譯「輸入維度」的 T_prev 是不同方向的矩陣乘法（左乘而非右乘）。

    average_with_parent：預設 True，維持原本行為——深層分支寫入時跟「對方自己那格的
    交叉通道」取 0.5 平均。設 False 時深層分支改成直接寫入 OT 翻譯值，不平均
    （align_first_layer=True 那一支本來就不平均，不受這個參數影響）。

    alpha：預設 None，維持原本行為（align_first_layer/average_with_parent 決定的邏輯
    不變）。設成 0~1 之間的數字時，不論 align_first_layer 或 average_with_parent 是什麼，
    兩支分支一律改成跟「child 呼叫前既有的交叉通道值」取加權平均：
    alpha*OT翻譯值 + (1-alpha)*child既有值。alpha=1 等同完整套用 OT，alpha=0 這一步
    完全不改動 W（退化成跟 distill_only 一樣，只剩後續蒸餾在動權重）——用來讓外層
    呼叫端把 alpha 隨 round/layer 衰減，讓 OT 的介入強度慢慢退場。
    """
    layer_names = _fusable_layer_names(child_policy)
    if stage_idx >= len(layer_names):
        return None

    dad_modules   = dict(dad_policy.named_modules())
    mom_modules   = dict(mom_policy.named_modules())
    child_modules = dict(child_policy.named_modules())
    child_dev = next(child_policy.parameters()).device  # dad/mom_policy 可能跟 child 不在同一裝置

    def _branch_of(name: str) -> str:
        # policy_net、value_net 是平行分支（共用同一份 feature extractor 輸出，
        # 不是接續關係），T_prev 只能在同一分支內鏈式傳遞，跨分支要重置。
        for b in ("policy_net", "value_net"):
            if b in name:
                return b
        return name

    T_prev = None  # None = identity（第一層共用觀測空間，或剛跨到新分支）
    prev_branch = None
    for i, name in enumerate(layer_names):
        branch = _branch_of(name)
        if branch != prev_branch:
            T_prev = None  # 跨分支，重置鏈式傳遞
        prev_branch = branch

        Wd = dad_modules[name].weight.data.to(child_dev)
        Wm = mom_modules[name].weight.data.to(child_dev)
        oh, ih = Wd.shape[0] // 2, Wd.shape[1] // 2
        Wd_pure = Wd[:oh, :ih]
        Wm_pure = Wm[oh:, ih:]

        if i == stage_idx:
            W = child_modules[name].weight.data
            # 只覆寫前 k 列（輸出神經元），其餘列維持原值（例如蒸餾基底）
            k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
            if k == 0:
                print(f"  [partial OT] ot_frac={ot_frac} → k=0，層 {name} 交叉通道完全不覆寫")
                return name
            if k < oh:
                print(f"  [partial OT] 層 {name}：只覆寫交叉通道前 {k}/{oh} 列，其餘維持原值")
            # alpha 模式先存 child 呼叫前既有值，當作衰減終點（alpha=0 時原封不動保留）
            prev_dad = W[:k, ih:].clone().float() if alpha is not None else None
            prev_mom = W[oh:oh + k, :ih].clone().float() if alpha is not None else None

            if T_prev is None and not align_first_layer:
                # 第一層：精確計算（T=I，不算 OT）
                W[:k, ih:] = Wd_pure[:k].clone()
                W[oh:oh + k, :ih] = Wm_pure[:k].clone()
            elif T_prev is None and align_first_layer:
                # 第一層也算 OT：比對 dad/mom 自己 oh 個隱藏神經元「誰的功能最像誰」，
                # 用配對結果 T 把兩邊神經元順序互相翻譯過去再寫入（左乘，作用在輸出/神經元維度）
                T_self = _compute_layer_transport(Wd_pure, Wm_pure)  # (oh, oh)
                aligned_dad = T_self.to(Wd_pure.device) @ Wd_pure.float()      # dad 翻譯成 mom 的神經元排列
                aligned_mom = T_self.t().to(Wm_pure.device) @ Wm_pure.float()  # mom 翻譯成 dad 的神經元排列
                if alpha is None:
                    W[:k, ih:] = aligned_dad[:k].to(W.dtype)
                    W[oh:oh + k, :ih] = aligned_mom[:k].to(W.dtype)
                else:
                    W[:k, ih:] = (alpha * aligned_dad[:k] + (1 - alpha) * prev_dad).to(W.dtype)
                    W[oh:oh + k, :ih] = (alpha * aligned_mom[:k] + (1 - alpha) * prev_mom).to(W.dtype)
            else:
                aligned_dad = (Wd_pure.float() @ T_prev.to(Wd_pure.device))
                aligned_mom = (Wm_pure.float() @ T_prev.t().to(Wm_pure.device))
                if alpha is not None:
                    # alpha 模式：跟 child 呼叫前既有值取加權平均，alpha=0 這一步等於沒做事
                    W[:k, ih:] = (alpha * aligned_dad[:k] + (1 - alpha) * prev_dad).to(W.dtype)
                    W[oh:oh + k, :ih] = (alpha * aligned_mom[:k] + (1 - alpha) * prev_mom).to(W.dtype)
                elif average_with_parent:
                    # 深層：交叉通道 = 0.5 * (OT 翻譯值 + 對方自己那格的交叉通道)
                    W[:k, ih:] = (0.5 * (aligned_dad[:k] + Wm[:k, ih:].float())).to(W.dtype)
                    W[oh:oh + k, :ih] = (0.5 * (aligned_mom[:k] + Wd[oh:oh + k, :ih].float())).to(W.dtype)
                else:
                    # 不平均：直接寫入 OT 翻譯值
                    W[:k, ih:] = aligned_dad[:k].to(W.dtype)
                    W[oh:oh + k, :ih] = aligned_mom[:k].to(W.dtype)
            return name

        T_prev = _compute_layer_transport(Wd_pure, Wm_pure)
    return None


@torch.no_grad()
def _iterative_refine_single_layer(
    child_policy: nn.Module,
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    stage_idx: int,
    ot_frac: float = 1.0,
    alpha: float = 0.5,
) -> Optional[str]:
    """
    漸進式 Iterative OT 的「重算」步驟：只重新對齊第 stage_idx 層。

    2026-09-07 實驗：不分 stage，全部層統一用「列對列」公式（原本只有 stage_idx==0
    這樣做，深層是用轉置去對齊欄，見 git 歷史）。理由：Wm_pure 的欄，不管在哪一層，
    天生就對得上 child 另一半當下實際收到的輸入——因為 child 建構(create_dual_channel_
    policy)是逐層整段複製 mom 的權重列，mom 那半邊網路在每一層都是自我一致、原封不動
    複製過去的，所以 Wm_pure 的欄位語意永遠等於「mom 那半邊上一層真正的輸出」，不需要
    再靠 T 做欄位翻譯。X_md_prev(dad 側交叉通道)的列＝dad 自己 oh 個神經元(身分已知、
    從未打亂)，比對基礎統一在 Wm_pure 也讀得到的同一組輸入上——第一層是共享的 obs
    維度，深層是複製過去的 mom 隱藏向量，兩種情況欄位語意都一致，所以同一條公式可以
    直接套用不必分支。X_dm_prev 同理對齊 Wd_pure。

    這跟舊版深層公式（保留 dad 自己的權重「樣式」，用轉置去猜哪個欄位對應到哪個 mom
    神經元）是兩種不同但都說得通的策略，哪個更好是實證問題，不是本次改動要下的結論。

    ⚠️ 這條公式不能把來源矩陣換成「X_md 用 Wd_pure、X_dm 用 Wm_pure」（曾經改過，
    commit b4dd545，已還原）。理由：第一層的輸入是**共享的 24 維 observation**，欄的
    左右半邊是 obs[0:12](本體感覺) 跟 obs[12:24](knee2/觸地 + 10 條 LIDAR)兩組不同的
    觀測特徵。X_md 是乘在 obs[12:24] 上的，來源就必須同樣是讀 obs[12:24] 的權重
    (Wm_pure)；換成 Wd_pure 等於把本體感覺調出來的權重拿去乘 LIDAR，第一層的地形感知
    直接壞掉——實測 child 的 LIDAR 權重範數暴增到父母的 2.2 倍，且表現隨地形難度單調
    崩潰(難度 0.0 尚有 301，難度 1.0 掉到 -5)。至於「X_md 的列要屬於 dad 神經元空間」
    這件事，是由 T 的列重排負責的(T @ Wm_pure 會把 mom 的列重排成 dad 的順序)，不需要
    也不能靠換來源矩陣達成。

    最後跟目前訓練值取加權平均（alpha 是 OT 目標值的權重，1-alpha 是保留
    目前訓練值的權重），而不是直接覆寫掉訓練成果。alpha=1 等同完整套用 OT 目標值，
    alpha=0 這一步等於完全不做任何事（W 維持呼叫前的值不變），退化成跟
    distill_only（完全不做 OT，只用蒸餾）一樣的效果——外層呼叫端可以讓 alpha
    隨 round 衰減，讓 OT 的介入強度隨著這層逐漸收斂而慢慢退場。
    """
    layer_names = _fusable_layer_names(child_policy)
    if stage_idx >= len(layer_names):
        return None
    name = layer_names[stage_idx]

    dad_m = dict(dad_policy.named_modules())[name]
    mom_m = dict(mom_policy.named_modules())[name]
    child_m = dict(child_policy.named_modules())[name]

    W = child_m.weight.data
    oh, ih = W.shape[0] // 2, W.shape[1] // 2
    if oh == 0 or ih == 0:
        return None

    dev = W.device
    # dad_policy/mom_policy 可能跟 child_policy 不在同一個裝置上，
    # _compute_layer_transport 要求兩個引數在同一裝置，這裡先統一搬到 child 的裝置
    Wd_pure = dad_m.weight.data[:oh, :ih].to(dev)
    Wm_pure = mom_m.weight.data[oh:, ih:].to(dev)

    X_md_prev = W[:oh, ih:].clone()  # (oh, ih)，剛被 PPO 訓練過的 dad 側交叉通道
    X_dm_prev = W[oh:, :ih].clone()  # (oh, ih)，剛被 PPO 訓練過的 mom 側交叉通道

    # 不分 stage，統一用「列對列」：重新配對「dad 的哪個神經元，功能上像 mom 的哪個
    # 神經元」，兩邊都拿讀同一組輸入的權重來比（X_md_prev 跟 Wm_pure 都讀得到 mom
    # 那半邊當下實際的輸入；X_dm_prev 跟 Wd_pure 同理讀 dad 那半邊的輸入）。
    T_self_md = _compute_layer_transport(X_md_prev, Wm_pure)  # (oh, oh)：列＝dad神經元(依訓練)，欄＝mom自己神經元
    T_self_dm = _compute_layer_transport(X_dm_prev, Wd_pure)  # (oh, oh)：列＝mom神經元(依訓練)，欄＝dad自己神經元

    translated_dad = (T_self_md.to(dev) @ Wm_pure.float())  # (oh, ih)，把 mom 自己的權重列翻譯成 dad 神經元順序
    translated_mom = (T_self_dm.to(dev) @ Wd_pure.float())  # (oh, ih)，把 dad 自己的權重列翻譯成 mom 神經元順序

    # 跟目前訓練值取加權平均，而不是直接覆寫——alpha 控制 OT 目標值介入的強度，
    # alpha=0 時 X_*_new 完全等於 X_*_prev（這一步等於沒做事，retains 訓練值）
    X_md_new = ((1 - alpha) * X_md_prev.float() + alpha * translated_dad).to(W.dtype)
    X_dm_new = ((1 - alpha) * X_dm_prev.float() + alpha * translated_mom).to(W.dtype)

    k = oh if ot_frac >= 1.0 else max(0, min(oh, int(round(oh * ot_frac))))
    if k > 0:
        W[:k, ih:] = X_md_new[:k]
        W[oh:oh + k, :ih] = X_dm_new[:k]
    return name


class _CrosstalkAlpha:
    """可變的退火狀態容器：alpha=0 代表交叉通道完全靜音，alpha=1 代表全開。"""
    def __init__(self):
        self.value: float = 0.0


def attach_crosstalk_annealing_hooks(
    policy: nn.Module,
    layer_names: List[str],
    baselines: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, "_CrosstalkAlpha"]:
    """
    幫 layer_names 中每一層的『交叉通道貢獻』掛上可調的退火係數 alpha（monkey-patch forward）。
    純通道權重完全不變。

    baselines 沒給：交叉通道貢獻 = 目前權重 * alpha（alpha=0 時完全靜音）。
    baselines 有給 {層名: 退火前的完整權重快照}：交叉通道貢獻 = baseline*(1-alpha) + 目前權重*alpha，
    也就是從 baseline（例如蒸餾基底）平滑插值到目前權重（例如 OT 對齊值），而不是從靜音升上來——
    避免把 baseline 已經校準好的交叉通道在 alpha=0 那一刻直接歸零。
    回傳 {層名: alpha容器}，呼叫端於訓練過程中自行更新 alpha_box.value（建議 0→1 線性升）。
    """
    alphas: Dict[str, _CrosstalkAlpha] = {}
    named = dict(policy.named_modules())
    for name in layer_names:
        layer = named.get(name)
        if layer is None or not isinstance(layer, nn.Linear):
            continue
        W = layer.weight
        oh, ih = W.shape[0] // 2, W.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        alpha_box = _CrosstalkAlpha()
        alphas[name] = alpha_box
        base_W = baselines.get(name) if baselines else None
        if base_W is not None:
            base_W = base_W.clone().to(W.device)

        def make_forward(orig_layer=layer, oh=oh, ih=ih, alpha_box=alpha_box, base_W=base_W):
            def forward(x):
                a = alpha_box.value
                W = orig_layer.weight
                if a >= 0.999:
                    return nn.functional.linear(x, W, orig_layer.bias)
                W_eff = W.clone()
                if base_W is None:
                    W_eff[:oh, ih:] = W_eff[:oh, ih:] * a   # 交叉通道 X_md：靜音 -> 目前權重
                    W_eff[oh:, :ih] = W_eff[oh:, :ih] * a   # 交叉通道 X_dm：靜音 -> 目前權重
                else:
                    W_eff[:oh, ih:] = base_W[:oh, ih:] * (1 - a) + W[:oh, ih:] * a  # baseline -> 目前權重
                    W_eff[oh:, :ih] = base_W[oh:, :ih] * (1 - a) + W[oh:, :ih] * a
                return nn.functional.linear(x, W_eff, orig_layer.bias)
            return forward

        layer.forward = make_forward()
    return alphas


def detach_crosstalk_annealing_hooks(policy: nn.Module, layer_names: List[str]):
    """移除退火 forward patch，還原成原生 nn.Linear.forward（權重數值不受影響）。"""
    named = dict(policy.named_modules())
    for name in layer_names:
        layer = named.get(name)
        if layer is not None and "forward" in layer.__dict__:
            del layer.forward


class CrosstalkAnnealCallback(BaseCallback):
    """
    訓練期間把交叉通道的 alpha 從 0 線性退火升到 1（跨 total_anneal_steps 步）。
    到達 1 之後維持在 1，讓交叉通道保持全開直到這個 stage 訓練結束。
    """
    def __init__(self, alpha_boxes: Dict[str, "_CrosstalkAlpha"], total_anneal_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.alpha_boxes = alpha_boxes
        self.total_anneal_steps = max(1, total_anneal_steps)
        self._start_step: Optional[int] = None

    def _on_training_start(self) -> None:
        self._start_step = self.num_timesteps

    def _on_step(self) -> bool:
        t = self.num_timesteps - self._start_step
        alpha = min(1.0, t / self.total_anneal_steps)
        for box in self.alpha_boxes.values():
            box.value = alpha
        return True


def distill_crosstalk_baseline(
    ppo_model: "PPO",
    pt_path: str,
    epochs: int = 5,
    lr: float = 0.008,
    batch_size: int = 4096,
    device: str = "cuda",
    exclude_layers: Optional[List[str]] = None,
    zero_init: bool = True,
    preloaded_blob: Optional[dict] = None,
):
    """
    交叉通道（X_md、X_dm）離線蒸餾，參數設定跟原本 zero_initialize_crosstalk 搭配的
    那組正牌流程一樣：zero_initialize_crosstalk 歸零 → freeze_pure_channels(bias_mode="freeze",
    allow_train_if_unsplit=False) 凍結純通道只留交叉通道可訓練 → Adam(lr=0.008)。

    exclude_layers：額外指定要凍結、不參與這次訓練的 fusable 層名，用在「某一層已經
    用 OT 對齊過、要保持固定」的情境——讓其餘層的交叉通道去適應這個已經固定的層。

    zero_init：預設 True，維持原本行為（先歸零交叉通道再蒸餾，當作全新的基底）。
    設 False 時跳過歸零，直接凍結純通道、對「目前已有的交叉通道值」（例如剛被 OT
    對齊過的值）做蒸餾微調——用在 progressive_iterative_ot_evolve 每輪 OT 之後，
    蒸餾只是要讓網路消化這次擾動，不該把 OT 剛寫入的值洗掉重來。

    preloaded_blob：傳入已經讀好的蒸餾資料 dict 就跳過 torch.load，避免同一份資料
    被重複從硬碟讀很多次（例如 progressive_iterative_ot_evolve 每輪都呼叫這個函式，
    20 次都重新讀一次 GB 等級的檔案會很慢，外面只讀一次、傳進來就好）。
    """
    policy = ppo_model.policy.to(device)
    named = dict(policy.named_modules())

    if zero_init:
        # zero_initialize_crosstalk 沒有排除機制，會把整個網路的交叉通道都歸零——
        # exclude_layers 指定的層（例如已經 OT 對齊過的第一層）先把權重存一份快照，
        # 歸零完再還原回去，避免它被誤蓋掉。
        exclude_snapshots = {}
        for name in (exclude_layers or []):
            m = named.get(name)
            if m is not None:
                exclude_snapshots[name] = m.weight.data.clone()

        zero_initialize_crosstalk(policy)

        for name, snap in exclude_snapshots.items():
            named[name].weight.data.copy_(snap)

    trainable_params, hook_handles = freeze_pure_channels(policy, bias_mode="freeze", allow_train_if_unsplit=False)

    # exclude_layers 指定的層要保持固定：把它的權重移出 trainable_params 並重新凍結
    for name in (exclude_layers or []):
        m = named.get(name)
        if m is None:
            continue
        m.weight.requires_grad = False
        trainable_params = [p for p in trainable_params if p is not m.weight]

    layer_names = _fusable_layer_names(policy)
    zero_msg = "已歸零並" if zero_init else "（沿用目前交叉通道值，未歸零）"
    print(f"[蒸餾交叉通道基底] {zero_msg}凍結純通道，"
          f"exclude_layers={exclude_layers or []}，共 {len(trainable_params)} 組可訓練參數")

    # ---- 以下訓練方法跟 train_network_offline 一致：KL 散度 + value MSE + auto-heuristic ----
    if preloaded_blob is not None:
        blob = preloaded_blob
    else:
        # map_location='cpu'：蒸餾資料檔案可能有 10GB+，直接載到 GPU 幾乎必定 OOM，
        # 訓練迴圈本來就會逐 batch 呼叫 .to(device)，資料留在 CPU 即可。
        blob = torch.load(pt_path, map_location="cpu")
    states = blob["states"].float()
    means = blob["teacher_means"].float()
    logstds_t_all = blob["teacher_logstds"].float()
    conf = blob["confidences"].float()
    values_t_all = blob["teacher_values"].float()

    logstd_vals = logstds_t_all.flatten().cpu().numpy()
    logstd_mean, logstd_std = np.mean(logstd_vals), np.std(logstd_vals)
    conf_std = conf.float().std().item()

    temperature = 1.5 + 0.3 * max(0, (-2.2 - logstd_mean))
    temperature = float(np.clip(temperature, 1.0, 2.3))
    alpha = 0.6 if conf_std > 0.2 else 0.7
    # 注意：跟 train_network_offline 一樣，auto-heuristic 算出來的 lr 只是參考顯示，
    # 實際 optimizer 用的是呼叫端指定的 lr（跟 zero_initialize_crosstalk 那組正牌流程
    # 一致的 0.008），不要拿這個自動算出來的值覆蓋掉。
    lr_scale = (1.0 - 0.5 * np.tanh(logstd_std + conf_std))
    auto_lr_display = float(np.clip(lr * lr_scale, 5e-5, 2e-4))
    print(f"  🧮 Auto-Heuristic Init: temp={temperature:.2f}, α={alpha:.2f}, (參考)auto_lr={auto_lr_display:.2e}, 實際 lr={lr:.4f}")

    top_idx = conf.argmax(dim=1)
    row = torch.arange(conf.shape[0], device=conf.device)
    mu_tgt = means[row, top_idx]
    logstd_tgt = torch.clamp(logstds_t_all[row, top_idx], min=-2.5, max=1.0)
    value_tgt = values_t_all[row, top_idx]

    ds = torch.utils.data.TensorDataset(states, mu_tgt, logstd_tgt, value_tgt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(trainable_params, lr=lr)

    def get_gaussian(obs_batch):
        f = policy.extract_features(obs_batch)
        lat_pi, lat_vf = policy.mlp_extractor(f) if policy.share_features_extractor \
            else policy.mlp_extractor(f, None)
        mu = policy.action_net(lat_pi)
        log_std = policy.log_std.expand_as(mu) if hasattr(policy, "log_std") else torch.zeros_like(mu)
        value = policy.value_net(lat_vf)
        return mu, log_std, value

    kl_target_min, kl_target_max, vf_coef = 20, 80, 0.5

    policy.train()
    with torch.enable_grad():
        for ep in range(epochs):
            run_pi_l, run_vf_l, n = 0.0, 0.0, 0
            for obs, mu_t, logstds_t, values_t in dl:
                obs, mu_t, logstds_t, values_t = obs.to(device), mu_t.to(device), logstds_t.to(device), values_t.to(device)
                opt.zero_grad(set_to_none=True)
                mu_s, logstds_s, values_s = get_gaussian(obs)

                var_s = torch.exp(2 * logstds_s)
                logstds_t_soft = logstds_t
                if temperature > 1.0:
                    logstds_t_soft = logstds_t + torch.log(torch.tensor(temperature, device=device))
                var_t = torch.exp(2 * logstds_t_soft)

                kl_div = (logstds_s - logstds_t_soft) + (var_t + (mu_t - mu_s).pow(2)) / (2 * var_s) - 0.5
                loss_soft = kl_div.sum(dim=-1).mean()
                loss_hard = torch.nn.functional.mse_loss(mu_s, mu_t)
                loss_pi = alpha * loss_soft + (1 - alpha) * loss_hard
                loss_vf = torch.nn.functional.mse_loss(values_s.squeeze(), values_t)
                loss = loss_pi + vf_coef * loss_vf

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                run_pi_l += loss_pi.item() * obs.size(0)
                run_vf_l += loss_vf.item() * obs.size(0)
                n += obs.size(0)

            avg_pi_loss, avg_vf_loss = run_pi_l / n, run_vf_l / n
            if avg_pi_loss > kl_target_max:
                temperature *= 1.15
                alpha = max(0.5, alpha - 0.05)
                print(f"  🔥 KL高 {avg_pi_loss:.1f} → temp={temperature:.2f}, α={alpha:.2f}")
            elif avg_pi_loss < kl_target_min:
                temperature *= 0.9
                alpha = min(0.8, alpha + 0.05)
                print(f"  🧊 KL低 {avg_pi_loss:.1f} → temp={temperature:.2f}, α={alpha:.2f}")
            print(f"  [蒸餾交叉通道基底] epoch {ep + 1}/{epochs}  KL={avg_pi_loss:.3f}  Value_MSE={avg_vf_loss:.3f}")

    policy.eval()
    # 移除梯度遮罩 hook，還原成全部可訓練：後面 progressive_recursive_ot_evolve
    # 需要整個網路可訓練才能重建 optimizer、做真實 PPO 訓練。
    for h in hook_handles:
        h.remove()
    for p in policy.parameters():
        p.requires_grad = True
    print("[蒸餾交叉通道基底] 完成，交叉通道已有蒸餾基底，接下來交給漸進式 OT 融合。")


def ot_first_layer_then_distill_rest(
    ppo_model: "PPO",
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    pt_path: str,
    epochs: int = 5,
    lr: float = 0.008,
    batch_size: int = 4096,
    device: str = "cuda",
    ot_frac: float = 1.0,
):
    """
    跟 distill_crosstalk_baseline（先蒸餾全部、再逐層退火 OT）順序相反：
      1. 只對第一個 fusable 層（例如 policy_net.0）做 OT 對齊。這層的輸入是原始觀測值，
         dad、mom 都還沒被混合過，OT 的神經元配對在這裡才有真正的統計意義（跟後面深層
         「輸入已經是混合過的隱藏向量、OT 配對可信度較低」不一樣）。
      2. 把這層凍結固定住（不參與後續訓練）。
      3. 其餘 fusable 層（例如 policy_net.2、value_net.0/.2）改用蒸餾（梯度下降，直接
         對真實 teacher 行為優化），讓它們去適應這個已經固定的第一層，而不是繼續用 OT
         硬猜一個沒有訓練依據的配對。
    回傳被 OT 固定的第一層層名（可能是 None，如果沒有任何 fusable 層）。
    """
    policy = ppo_model.policy.to(device)
    layer_names = _fusable_layer_names(policy)
    if not layer_names:
        print("[OT首層+蒸餾其餘] 沒有可融合的層，跳過。")
        return None

    first_layer = layer_names[0]
    rest_layers = layer_names[1:]

    if ot_frac < 1.0:
        # 部分 OT：先把第一層的交叉通道整塊歸零（小噪音），再讓 OT 只覆寫前 ot_frac 比例的列。
        # 沒被 OT 覆寫的列會維持在近乎靜音的狀態，而不是留著 dad/mom 原始未對齊的值。
        named_first = dict(policy.named_modules())[first_layer]
        Wf = named_first.weight.data
        oh_f, ih_f = Wf.shape[0] // 2, Wf.shape[1] // 2
        nn.init.normal_(Wf[:oh_f, ih_f:], mean=0.0, std=0.001)
        nn.init.normal_(Wf[oh_f:, :ih_f], mean=0.0, std=0.001)

    recursive_ot_fuse_single_layer(policy, dad_policy, mom_policy, 0, ot_frac=ot_frac)
    print(f"[OT首層+蒸餾其餘] 已對第一層 {first_layer} 做 OT 對齊（ot_frac={ot_frac}），這層接下來會凍結固定")

    if rest_layers:
        distill_crosstalk_baseline(
            ppo_model, pt_path, epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, exclude_layers=[first_layer],
        )
    else:
        print("[OT首層+蒸餾其餘] 只有一層可融合，沒有其餘層需要蒸餾。")

    return first_layer


def progressive_recursive_ot_evolve(
    child_agent: "PPO",
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    env=None,
    steps_per_stage: int = 20_000,
    max_stages: Optional[int] = None,
    anneal: bool = True,
    ot_frac: float = 1.0,
):
    """
    漸進式逐層 Recursive OT 融合：
      1. 尚未輪到的層維持 create_dual_channel_policy 拼接時的原始值（dad/mom 各自完整一行，
         含他們自己上一代融合繼承來的交叉通道）——不歸零。dad/mom 本身可能就是融合過的個體，
         他們自己那份交叉通道對他們自己半邊的功能是有意義的，歸零會在還沒開始這一輪融合前
         就先破壞掉這個功能（這正是先前 --test_ties 測出 child 一開始就崩潰的根因）。
      2. stage k：用 recursive OT 把第 k 層的交叉通道覆寫成這一輪的 OT 翻譯值（純通道不動、
         不縮放）。
         anneal=True（預設）：幫這一層掛上退火 hook，交叉通道從「這層原本的 baseline
         （蒸餾基底或上一輪殘留值）」在 steps_per_stage 步內平滑插值到「完整 OT 值」，
         期間用 PPO 正常訓練全網（不凍結）。
         anneal=False：不掛退火 hook，這一層從 stage 一開始就是完整 OT 值（沒有漸進混合
         過程），直接在這個狀態下訓練 steps_per_stage 步，用來對照「有沒有退火」的差異。
      3. alpha 升到 1（或 anneal=False 時從頭就是滿的）後，這層權重就是完整 OT 值
      4. 全部層融合完後，交還呼叫端做最終微調

    若傳入 env，訓練期間會掛上 AutoDifficultyCallback（跟 dad/mom 原本訓練時同一套
    難度自動升級 + hard_seeds 難度池機制），而不是只在固定難度下訓練。

    zero_initialize_crosstalk 僅用於舊版「歸零 + 離線蒸餾」流程，漸進式退火不應呼叫它。
    """
    policy = child_agent.policy
    layer_names = _fusable_layer_names(policy)
    n_layers = len(layer_names)
    n_stages_to_run = n_layers if max_stages is None else min(max_stages, n_layers)
    print(f"[Progressive OT] 共 {n_layers} 層可融合，這次只跑前 {n_stages_to_run} 層，每 stage 訓練 {steps_per_stage} 步，anneal={anneal}")

    _rebuild_policy_optimizer(policy)

    auto_difficulty_callback = None
    if env is not None:
        auto_difficulty_callback = AutoDifficultyCallback(
            env, None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,
            shared_flags=None, cooldown_steps=0, hardseed_save_path="./logs/hard_seeds.json",
        )
        print("[Progressive OT] 已接上 AutoDifficultyCallback（難度自動升級 + hard_seeds 難度池）")

    for stage in range(n_stages_to_run):
        # 覆寫成 OT 值之前，先把這層目前的權重（蒸餾基底，或上一輪殘留值）存一份，
        # anneal=True 時退火才能從這個 baseline 平滑插值到新的 OT 值，而不是從靜音(0)開始。
        stage_layer_name = layer_names[stage]
        baseline_W = dict(policy.named_modules())[stage_layer_name].weight.data.clone()

        fused_name = recursive_ot_fuse_single_layer(policy, dad_policy, mom_policy, stage, ot_frac=ot_frac)

        if steps_per_stage > 0 and anneal:
            print(f"\n[Progressive OT] stage {stage+1}/{n_layers}：已寫入層 {fused_name} 的 OT 交叉通道，開始退火")
            alpha_boxes = attach_crosstalk_annealing_hooks(policy, [fused_name], baselines={fused_name: baseline_W})
            anneal_cb = CrosstalkAnnealCallback(alpha_boxes, total_anneal_steps=steps_per_stage)
            callbacks = [anneal_cb] + ([auto_difficulty_callback] if auto_difficulty_callback is not None else [])
            child_agent.learn(total_timesteps=steps_per_stage, callback=callbacks, progress_bar=True)
            detach_crosstalk_annealing_hooks(policy, [fused_name])
        elif steps_per_stage > 0:
            print(f"\n[Progressive OT] stage {stage+1}/{n_layers}：已寫入層 {fused_name} 的 OT 交叉通道（無退火，直接生效），開始訓練")
            callbacks = [auto_difficulty_callback] if auto_difficulty_callback is not None else []
            child_agent.learn(total_timesteps=steps_per_stage, callback=callbacks, progress_bar=True)

    if n_stages_to_run < n_layers:
        remaining = layer_names[n_stages_to_run:]
        print(f"[Progressive OT] 已跑完前 {n_stages_to_run} 層，剩下 {len(remaining)} 層維持蒸餾基底值，不做 OT：{remaining}")
    else:
        print("[Progressive OT] 全部層融合完成。")


def _pre_finetune_out_path(args) -> Optional[str]:
    """
    推導「PPO 微調前」那份初始化模型要存去哪。
    明確給了 --progressive_pre_finetune_out 就用它（空字串＝停用）；
    沒給的話，從最終輸出路徑衍生：xxx.pkl → xxx_preppo.pkl。
    """
    explicit = getattr(args, "progressive_pre_finetune_out", None)
    if explicit is not None:
        return explicit or None
    final_path = getattr(args, "ties_out", None) or f"./models/ties_test_child_{args.crossover}.pkl"
    base, ext = os.path.splitext(final_path)
    return f"{base}_preppo{ext or '.pkl'}"


def progressive_iterative_ot_evolve(
    child_agent: "PPO",
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    distill_pt: str,
    env=None,
    n_rounds: int = 10,
    distill_epochs: int = 5,
    distill_lr: float = 0.008,
    final_finetune_steps: int = 1_000_000,
    max_stages: Optional[int] = None,
    ot_frac: float = 1.0,
    device: str = "cuda",
    alpha_init: float = 1.0,
    alpha_gamma: float = 0.7169,
    pre_finetune_save_path: Optional[str] = None,
):
    """
    漸進式 Iterative OT 融合：外層逐層(layer)，內層逐輪(round)。

    pre_finetune_save_path：給了路徑的話，在最終 PPO 微調「開始之前」先把 policy 存一份，
    留下純 OT+蒸餾初始化的結果，方便跟微調後的版本對照（分離「初始化品質」與「PPO 修復
    能力」兩個因素）。final_finetune_steps=0 時不會重複存，因為那時最終權重本來就等同
    微調前的權重，直接由呼叫端的存檔負責。

    alpha_init / alpha_gamma：每個 stage 的 round 迴圈裡，第 round_idx 輪（0-based，
    每換一層歸零重新從 alpha_init 開始)用的 OT 介入強度是
    alpha_t = alpha_init * (alpha_gamma ** round_idx)，隨 round 指數衰減。alpha_t=1
    等同完整套用 OT 翻譯值，alpha_t=0 那一輪的 OT 步驟完全不改動權重，退化成跟
    distill_only（不做 OT，只用蒸餾）一樣的效果——只剩該輪的 distill_crosstalk_baseline
    在調整權重。預設 alpha_init=1.0（第一輪大力對齊，完整套用 OT）、alpha_gamma=0.7169
    （搭配預設 n_rounds=10，十輪的 alpha_t ≈ 1.0, 0.717, 0.514, 0.368, 0.264, 0.189,
    0.136, 0.097, 0.070, 0.050——最後一輪幾乎完全停止干預，交由訓練自己收斂）。若調整
    n_rounds，需重新用 gamma=(alpha_target/alpha_init)**(1/(n_rounds-1)) 反推 gamma，
    否則衰減速度會跟輪數對不上。

    每個 stage（第 L 層）：
      round 1：用 dad/mom 固定純通道權重算 T，寫入交叉通道（recursive_ot_fuse_single_layer，
        align_first_layer=True——連第一層也真的算一次 OT，不用 T=I 精確複製；
        alpha=alpha_t——OT 翻譯值跟 child 呼叫前既有的交叉通道值取加權平均，取代原本
        寫死的 average_with_parent 0.5 平均）。
      round 2..n_rounds：用「這一層剛被蒸餾訓練過」的交叉通道權重當比對目標重新算 T，
        翻譯出對應的值後跟目前訓練值取平均（_iterative_refine_single_layer）——重算結果
        會隨訓練改變，不是每輪都寫回同一個值。
          - 深層(stage>=1)：T 翻譯的是 dad/mom 純通道，欄位順序永遠鎖定在 dad/mom
            自己的神經元順序上，不會像置換訓練值欄位那樣把權重跟下一層實際讀到的
            訊號斷開。
          - 第一層(stage 0)：輸入端(觀測值)沒有排列可對齊，但改成重新配對「dad/mom
            自己的 oh 個神經元」彼此的功能對應，比對基礎統一在同一組觀測維度上
            （不像 round 1 的 T_self 是跨兩組不同觀測維度硬比）。
      每輪 OT 之後，用蒸餾（凍結純通道，distill_epochs 個 epoch，distill_lr 學習率）
        消化這次擾動，而不是丟進真實環境跑 PPO——蒸餾是對著離線資料做監督式梯度下降，
        不用蒐集 rollout、也不會被難度爬升打斷，比每輪都真實環境訓練快很多。
    這一層跑滿 n_rounds 輪才前進到下一層，重複整個流程直到 max_stages（或全部層）跑完。

    全部層、全部輪都蒸餾消化完之後，才接一次 final_finetune_steps 步的真實環境訓練
    （帶 AutoDifficultyCallback 難度爬升）——難度爬升造成的重新適應只發生這一次，
    不會跟每輪的 OT 擾動疊在一起、每次都要重新適應環境+難度。

    跟 progressive_recursive_ot_evolve 的差異：
      - 沒有 alpha 退火，每輪直接覆寫交叉通道。
      - 同一層會反覆「OT → 蒸餾」n_rounds 次，不是只做一次就換下一層。
    """
    policy = child_agent.policy
    layer_names = _fusable_layer_names(policy)
    n_layers = len(layer_names)
    n_stages_to_run = n_layers if max_stages is None else min(max_stages, n_layers)
    print(f"[Progressive Iterative OT] 共 {n_layers} 層可融合，這次跑前 {n_stages_to_run} 層，"
          f"每層 {n_rounds} 輪，每輪用蒸餾（凍結純通道，{distill_epochs} epoch）消化 OT 擾動")

    # 蒸餾資料只需要讀一次，之後每輪都重複使用，避免同一份檔案被反覆從硬碟讀取
    distill_blob = torch.load(distill_pt, map_location="cpu")

    for stage in range(n_stages_to_run):
        for round_idx in range(n_rounds):
            alpha_t = alpha_init * (alpha_gamma ** round_idx)
            fused_name = _iterative_refine_single_layer(policy, dad_policy, mom_policy, stage, ot_frac=ot_frac, alpha=alpha_t)

            if fused_name is None:
                print(f"[Progressive Iterative OT] stage {stage+1} 找不到對應層，跳過")
                break

            print(f"\n[Progressive Iterative OT] stage {stage+1}/{n_stages_to_run} 層 {fused_name}  "
                  f"round {round_idx+1}/{n_rounds}（alpha={alpha_t:.4f}）：OT 對齊完成（含初始交叉通道值），用蒸餾消化這次擾動")

            distill_crosstalk_baseline(
                child_agent, distill_pt, epochs=distill_epochs, lr=distill_lr,
                device=device, zero_init=False, preloaded_blob=distill_blob,
            )

    if n_stages_to_run < n_layers:
        remaining = layer_names[n_stages_to_run:]
        print(f"[Progressive Iterative OT] 已跑完前 {n_stages_to_run} 層，"
              f"剩下 {len(remaining)} 層維持原始拼接值，不做 OT：{remaining}")
    else:
        print("[Progressive Iterative OT] 全部層融合完成。")

    if pre_finetune_save_path and final_finetune_steps > 0:
        os.makedirs(os.path.dirname(pre_finetune_save_path) or ".", exist_ok=True)
        with open(pre_finetune_save_path, "wb") as f:
            cloudpickle.dump(policy, f)
        print(f"[Progressive Iterative OT] 已儲存「PPO 微調前」的純 OT+蒸餾初始化模型：{pre_finetune_save_path}")

    if final_finetune_steps > 0:
        _rebuild_policy_optimizer(policy)
        auto_difficulty_callback = None
        if env is not None:
            auto_difficulty_callback = AutoDifficultyCallback(
                env, None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,
                shared_flags=None, cooldown_steps=0, hardseed_save_path="./logs/hard_seeds.json",
            )
            print("[Progressive Iterative OT] 已接上 AutoDifficultyCallback（難度自動升級 + hard_seeds 難度池）")
        print(f"\n[Progressive Iterative OT] 全部層蒸餾消化完成，開始最終真實環境微調，共 {final_finetune_steps} 步")
        callbacks = [auto_difficulty_callback] if auto_difficulty_callback is not None else []
        child_agent.learn(total_timesteps=final_finetune_steps, callback=callbacks, progress_bar=True)


def progressive_iterative_ot_evolve_focused(
    child_agent: "PPO",
    dad_policy: nn.Module,
    mom_policy: nn.Module,
    distill_pt: str,
    env=None,
    n_rounds: int = 10,
    distill_epochs: int = 5,
    distill_lr: float = 0.008,
    final_finetune_steps: int = 1_000_000,
    max_stages: Optional[int] = None,
    ot_frac: float = 1.0,
    device: str = "cuda",
    alpha_init: float = 1.0,
    alpha_gamma: float = 0.7169,
    pre_finetune_save_path: Optional[str] = None,
):
    """
    跟 progressive_iterative_ot_evolve 的唯一差異，在於每輪蒸餾消化的「範圍」：

      - 開跑前：先對「全部」fusable 層的交叉通道蒸餾一次（zero_init=True，跟舊版
        一次性蒸餾基底做法相同），當作整個流程的起點。
      - 每個 stage 的每一輪：OT 對齊完之後，蒸餾消化只聚焦在「這一輪剛被 OT 對齊過
        的那一層」，其餘所有 fusable 層的交叉通道全部凍結（distill_crosstalk_baseline
        的 exclude_layers 傳入除了當下這層以外的全部層名）——用來測試「只讓 OT 剛動
        過的那一層被蒸餾照顧」跟 progressive_iterative_ot_evolve「全部層一起蒸餾」
        這兩種做法，效果差在哪裡。

    OT 對齊、round 1/round 2+ 的邏輯、最終真實環境微調，其餘部分跟
    progressive_iterative_ot_evolve 完全相同。
    """
    policy = child_agent.policy
    layer_names = _fusable_layer_names(policy)
    n_layers = len(layer_names)
    n_stages_to_run = n_layers if max_stages is None else min(max_stages, n_layers)
    print(f"[Progressive Iterative OT - Focused] 共 {n_layers} 層可融合，這次跑前 {n_stages_to_run} 層，"
          f"每層 {n_rounds} 輪，每輪只聚焦當下這一層做蒸餾（其餘層凍結）")

    # 蒸餾資料只需要讀一次，之後每次呼叫重複使用
    distill_blob = torch.load(distill_pt, map_location="cpu")

    print(f"\n[Progressive Iterative OT - Focused] 開跑前先蒸餾全部交叉通道一次（基底）")
    distill_crosstalk_baseline(
        child_agent, distill_pt, epochs=distill_epochs, lr=distill_lr,
        device=device, zero_init=True, preloaded_blob=distill_blob,
    )

    for stage in range(n_stages_to_run):
        for round_idx in range(n_rounds):
            alpha_t = alpha_init * (alpha_gamma ** round_idx)
            if round_idx == 0:
                fused_name = recursive_ot_fuse_single_layer(
                    policy, dad_policy, mom_policy, stage, ot_frac=ot_frac,
                    align_first_layer=True, alpha=alpha_t,
                )
            else:
                fused_name = _iterative_refine_single_layer(policy, dad_policy, mom_policy, stage, ot_frac=ot_frac, alpha=alpha_t)

            if fused_name is None:
                print(f"[Progressive Iterative OT - Focused] stage {stage+1} 找不到對應層，跳過")
                break

            other_layers = [n for n in layer_names if n != fused_name]
            print(f"\n[Progressive Iterative OT - Focused] stage {stage+1}/{n_stages_to_run} 層 {fused_name}  "
                  f"round {round_idx+1}/{n_rounds}（alpha={alpha_t:.4f}）：OT 對齊完成，只對這一層做蒸餾消化（其餘 {len(other_layers)} 層凍結）")

            distill_crosstalk_baseline(
                child_agent, distill_pt, epochs=distill_epochs, lr=distill_lr,
                device=device, zero_init=False, exclude_layers=other_layers,
                preloaded_blob=distill_blob,
            )

    if n_stages_to_run < n_layers:
        remaining = layer_names[n_stages_to_run:]
        print(f"[Progressive Iterative OT - Focused] 已跑完前 {n_stages_to_run} 層，"
              f"剩下 {len(remaining)} 層維持蒸餾基底值，不做 OT：{remaining}")
    else:
        print("[Progressive Iterative OT - Focused] 全部層融合完成。")

    if pre_finetune_save_path and final_finetune_steps > 0:
        os.makedirs(os.path.dirname(pre_finetune_save_path) or ".", exist_ok=True)
        with open(pre_finetune_save_path, "wb") as f:
            cloudpickle.dump(policy, f)
        print(f"[Progressive Iterative OT - Focused] 已儲存「PPO 微調前」的純 OT+蒸餾初始化模型：{pre_finetune_save_path}")

    if final_finetune_steps > 0:
        _rebuild_policy_optimizer(policy)
        auto_difficulty_callback = None
        if env is not None:
            auto_difficulty_callback = AutoDifficultyCallback(
                env, None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,
                shared_flags=None, cooldown_steps=0, hardseed_save_path="./logs/hard_seeds.json",
            )
            print("[Progressive Iterative OT - Focused] 已接上 AutoDifficultyCallback（難度自動升級 + hard_seeds 難度池）")
        print(f"\n[Progressive Iterative OT - Focused] 全部層蒸餾消化完成，開始最終真實環境微調，共 {final_finetune_steps} 步")
        callbacks = [auto_difficulty_callback] if auto_difficulty_callback is not None else []
        child_agent.learn(total_timesteps=final_finetune_steps, callback=callbacks, progress_bar=True)


@torch.no_grad()
def verify_zero_initialize_crosstalk(policy_before: nn.Module, policy_after: nn.Module, atol=1e-8):
    """
    驗證 zero_initialize_crosstalk() 是否僅修改 cross-block。
    檢查：
      - 父->父 (左上) 與 母->母 (右下) 權重區塊保持一致
      - cross-block (右上、左下) 被改動 (差異 > atol)
    """
    ok = True
    before_modules = dict(policy_before.named_modules())
    after_modules  = dict(policy_after.named_modules())

    for name, m_after in after_modules.items():
        if not isinstance(m_after, nn.Linear):
            continue
        m_before = before_modules.get(name)
        if m_before is None:
            continue

        Wb = m_before.weight.data.clone()
        Wa = m_after.weight.data.clone()

        if Wb.shape != Wa.shape or Wb.ndim != 2:
            continue

        out_half, in_half = Wa.shape[0] // 2, Wa.shape[1] // 2
        if out_half == 0 or in_half == 0:
            continue

        # 區塊定義
        dad_dad_block = Wa[:out_half, :in_half]   # 父->父
        mom_mom_block = Wa[out_half:, in_half:]   # 母->母
        cross_mf_block = Wa[:out_half, in_half:]  # 母->父
        cross_fm_block = Wa[out_half:, :in_half]  # 父->母

        # 檢查哪些區塊被改動
        diff_dad = (dad_dad_block - Wb[:out_half, :in_half]).abs().max().item()
        diff_mom = (mom_mom_block - Wb[out_half:, in_half:]).abs().max().item()
        diff_cross1 = (cross_mf_block - Wb[:out_half, in_half:]).abs().max().item()
        diff_cross2 = (cross_fm_block - Wb[out_half:, :in_half]).abs().max().item()

        pass_pure = diff_dad <= atol and diff_mom <= atol
        pass_cross = (diff_cross1 > atol*10) or (diff_cross2 > atol*10)

        if pass_pure and pass_cross:
            print(f"[OK] {name:40s} 父母區未改動, cross區已更新 ✅")
        else:
            ok = False
            print(f"[X] {name:40s} diff_dad={diff_dad:.2e}, diff_mom={diff_mom:.2e}, "
                  f"diff_cross1={diff_cross1:.2e}, diff_cross2={diff_cross2:.2e}")

    print("\n✅ 通過全部層驗證" if ok else "\n⚠️ 有層不符合預期")
    return ok

def _sd_to_vector(state_dict: dict) -> torch.Tensor:
    """把 state_dict 中所有參數按 key 排序後，拼成一個 1-D float32 向量。"""
    return torch.cat([
        v.detach().cpu().float().reshape(-1)
        for _, v in sorted(state_dict.items())
    ])

def _vector_to_sd(vector: torch.Tensor, ref_sd: dict) -> dict:
    """把向量還原成與 ref_sd 同形狀的 state_dict。"""
    new_sd, offset = {}, 0
    for k, v in sorted(ref_sd.items()):
        n = v.numel()
        new_sd[k] = vector[offset: offset + n].reshape(v.shape).to(v.dtype)
        offset += n
    return new_sd

def _topk_trim(M: torch.Tensor, k: float) -> torch.Tensor:
    """對矩陣 M（每列一個模型的參數向量），只保留每列 top-k% 絕對值，其餘歸零。"""
    if k >= 1.0:
        return M.clone()
    d = M.shape[1]
    keep = max(1, int(d * k))
    kth, _ = M.abs().kthvalue(d - keep + 1, dim=1, keepdim=True)
    return M * (M.abs() >= kth)

def _resolve_sign(M: torch.Tensor, method: str = "mass") -> torch.Tensor:
    """每個參數位置投票決定符號（mass / normfrac / normmass）。"""
    if method == "mass":
        sign = torch.sign(M.sum(dim=0))
    elif method == "normfrac":
        row_norms = torch.norm(M, dim=1, keepdim=True).clamp(min=1e-8)
        norm_fracs = (M ** 2) / (row_norms ** 2)
        sign = torch.sign(M[norm_fracs.argmax(dim=0), torch.arange(M.shape[1])])
    elif method == "normmass":
        row_norms = torch.norm(M, dim=1, keepdim=True).clamp(min=1e-8)
        norm_fracs = (M ** 2) / (row_norms ** 2)
        sign = (M.sign() * norm_fracs.abs()).sum(dim=0).sign()
    else:
        raise ValueError(f"未知的 resolve_method: {method}")
    majority = torch.sign(M.sum(dim=0))
    sign[sign == 0] = majority[sign == 0]
    sign[sign == 0] = 1.0
    return sign

def _disjoint_merge(M: torch.Tensor, sign: torch.Tensor, func: str = "mean") -> torch.Tensor:
    """只保留與 elected sign 一致的值，再做 mean/sum/max 聚合。"""
    agree_mask = torch.where(sign.unsqueeze(0) > 0, M > 0, M < 0)
    selected = M * agree_mask.float()
    if func == "mean":
        counts = agree_mask.float().sum(dim=0).clamp(min=1)
        return selected.sum(dim=0) / counts
    elif func == "sum":
        return selected.sum(dim=0)
    elif func == "max":
        result = selected.abs().max(dim=0)[0]
        return result * sign
    else:
        raise ValueError(f"未知的 merge_func: {func}")

@torch.no_grad()

def create_refined_differential_optimizer(
    policy: nn.Module, 
    lr_pure: float, 
    cross_lr_scale_factor: float = 0.01 # 將交叉學習率改為一個縮放因子
) -> torch.optim.Adam:
    """
    創建一個差異化學習率的優化器（精細版）。
    使用梯度掛鉤 (gradient hook) 來精準地縮放交叉通道的梯度。
    """
    print(f"創建精細版差異化優化器，基礎學習率: {lr_pure}, 交叉通道梯度縮放因子: {cross_lr_scale_factor}")
    
    # 步驟 1: 為所有參數設定一個統一的、較高的學習率
    # 我們不再需要手動分離參數組
    all_params = policy.parameters()
    optimizer = torch.optim.Adam(all_params, lr=lr_pure)

    # 步驟 2: 遍歷所有線性層的權重，並為它們註冊一個特製的掛鉤
    for name, module in policy.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight
            
            # 獲取維度資訊
            in_half = W.shape[1] // 2
            out_half = W.shape[0] // 2

            if in_half == 0 or out_half == 0:
                continue
            
            # --- 這是 Python 的一個技巧，稱為「閉包 (Closure)」---
            # 我們定義一個函式，它會「記住」當前層的維度和縮放因子
            # 然後返回我們真正需要的掛鉤函式
            def create_hook(oh=out_half, ih=in_half, sf=cross_lr_scale_factor):
                def hook(grad):
                    # 這是掛鉤的核心：修改梯度
                    with torch.no_grad():
                        # # 右上角 (母 -> 父) 的梯度，乘以縮放因子
                        # grad[:oh, ih:].mul_(sf)
                        # # 左下角 (父 -> 母) 的梯度，乘以縮放因子
                        # grad[oh:, :ih].mul_(sf)
                        grad[:oh, :ih].mul_(sf)
                        # 左下角 (父 -> 母) 的梯度，乘以縮放因子
                        grad[oh:, ih:].mul_(sf)
                        # grad[:oh, :ih].zero_()
                        # # 左下角 (父 -> 母) 的梯度，乘以縮放因子
                        # grad[oh:, ih:].zero_()
                    return grad
                return hook
            
            # 為當前權重張量 W 註冊這個新鮮創建的掛鉤
            W.register_hook(create_hook())

    print("所有權重矩陣的梯度掛鉤均已註冊。")
    return optimizer
def freeze_pure_channels(
    policy: nn.Module,
    *,
    bias_mode: str = "freeze",     # "freeze" | "train"
    allow_train_if_unsplit: bool = False  # in_half==0 or out_half==0 時是否整層可訓練
) -> Tuple[List[torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    凍結純淨通道（左上=父->父、右下=母->母），只允許『交叉通道』（右上、左下）學習。
    bias_mode:
      - "freeze": bias 也凍結（掛 hook 將梯度清零、且不加入 optimizer）
      - "train" : bias 允許學習（加入 optimizer）
    allow_train_if_unsplit:
      - True  : 對無法二分的層(oh==0 或 ih==0)整層可訓練
      - False : 這類層整層凍結
    回傳: (trainable_params, hook_handles)
    """
    print("--- 正在凍結純淨通道 (父->父, 母->母) ---")

    # 先關閉所有參數的 requires_grad
    for p in policy.parameters():
        p.requires_grad = False

    trainable_params: List[torch.Tensor] = []
    hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight
        oh = W.shape[0] // 2
        ih = W.shape[1] // 2

        # 無法二分的層
        if oh == 0 or ih == 0:
            if allow_train_if_unsplit:
                # 整層可訓練
                W.requires_grad = True
                trainable_params.append(W)
                if module.bias is not None and bias_mode == "train":
                    module.bias.requires_grad = True
                    trainable_params.append(module.bias)
            # else: 全凍結（不做任何事）
            continue

        # 權重：整張 weight 允許計算梯度，但用 hook 清掉純淨區塊梯度
        W.requires_grad = True
        def make_weight_hook(oh=oh, ih=ih):
            def hook(grad):
                # 清掉純淨塊：左上(父->父)、右下(母->母)
                grad[:oh, :ih] = 0.0
                grad[oh:, ih:] = 0.0
                return grad
            return hook
        h = W.register_hook(make_weight_hook())
        hook_handles.append(h)
        trainable_params.append(W)  # 交叉塊會有梯度 → 交給 optimizer

        # bias：依 bias_mode 決定是否訓練
        if module.bias is not None:
            B = module.bias
            if bias_mode == "freeze":
                # 凍結：掛 hook 把整個 bias 梯度清零（或只清純淨上/下半）
                B.requires_grad = True  # 需要 True 才會走 hook
                def make_bias_hook(oh=oh):
                    def hook(grad):
                        grad.zero_()         # 完全不更新 bias
                        # 若只想凍結純淨、放開交叉對應的 bias（通常不需要），可改成：
                        # grad[:oh] = 0.0
                        # grad[oh:] = 0.0
                        return grad
                    return hook
                hook_handles.append(B.register_hook(make_bias_hook()))
                # 不加入 trainable_params（雖然有 hook 也可以加入，但沒必要）
            else:  # "train"
                B.requires_grad = True
                trainable_params.append(B)

    # 去重（避免重複加入）
    trainable_params = list(dict.fromkeys(trainable_params))
    print(f"DEBUG: Found {len(trainable_params)} trainable parameters.")
    print(f"DEBUG: Trainable parameter shapes: {[tuple(p.shape) for p in trainable_params]}")
    return trainable_params, hook_handles

# ================================================================
# 核心工具二：解凍所有參數
# ================================================================
def snapshot_params(policy: nn.Module):
    """抓取每個 Linear 的 weight/bias 張量拷貝，供前後比對。"""
    snap = {}
    for name, m in policy.named_modules():
        if isinstance(m, nn.Linear):
            snap[f"{name}.weight"] = m.weight.detach().clone()
            if m.bias is not None:
                snap[f"{name}.bias"] = m.bias.detach().clone()
    return snap

@torch.no_grad()
def diff_report_before_after(policy_before: nn.Module, policy_after: nn.Module, atol=1e-12):
    ok = True
    before_modules = dict(policy_before.named_modules())
    after_modules  = dict(policy_after.named_modules())

    print("\n=== Freeze 驗證報告（純淨塊應 ≈ 0）===")
    for name, m_after in after_modules.items():
        if not isinstance(m_after, nn.Linear):
            continue
        m_before = before_modules.get(name)
        if m_before is None:
            continue

        # 關鍵：統一到 CPU
        Wa = m_after.weight.detach().cpu()
        Wb = m_before.weight.detach().cpu()

        if Wa.shape != Wb.shape or Wa.ndim != 2:
            continue

        oh, ih = Wa.shape[0] // 2, Wa.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        d_pp = (Wa[:oh, :ih] - Wb[:oh, :ih]).abs().max().item()   # 父->父 左上（應≈0）
        d_pm = (Wa[:oh, ih:] - Wb[:oh, ih:]).abs().max().item()   # 父->母 右上（允許變）
        d_mp = (Wa[oh:, :ih] - Wb[oh:, :ih]).abs().max().item()   # 母->父 左下（允許變）
        d_mm = (Wa[oh:, ih:] - Wb[oh:, ih:]).abs().max().item()   # 母->母 右下（應≈0）

        if (m_after.bias is not None) and (m_before.bias is not None):
            ba = m_after.bias.detach().cpu()
            bb = m_before.bias.detach().cpu()
            if ba.numel() == 2 * oh:
                d_bt = (ba[:oh] - bb[:oh]).abs().max().item()
                d_bb = (ba[oh:] - bb[oh:]).abs().max().item()
            else:
                d_bt = d_bb = float('nan')
        else:
            d_bt = d_bb = float('nan')

        pure_ok = (d_pp <= atol) and (d_mm <= atol) and (np.isnan(d_bt) or d_bt <= atol) and (np.isnan(d_bb) or d_bb <= atol)
        ok &= pure_ok

        print(f"{'[OK]' if pure_ok else '[X ]'} {name:38s} "
              f"pure_pp:{d_pp:.2e}  pure_mm:{d_mm:.2e}  "
              f"cross_pm:{d_pm:.2e}  cross_mp:{d_mp:.2e}  "
              f"bias_top:{d_bt:.2e}  bias_bot:{d_bb:.2e}")

    print("✅ 純淨塊皆未更新（凍結生效）" if ok else "⚠️ 發現純淨塊有更新（凍結未生效）")
    return ok
def list_optimizer_params(optimizer: torch.optim.Optimizer):
    """列出目前 optimizer 管轄的參數形狀與名稱數量（粗略檢查）"""
    total = 0
    print("\n--- Optimizer 參數概覽 ---")
    for gi, g in enumerate(optimizer.param_groups):
        print(f"Group {gi} lr={g.get('lr')}")
        for p in g['params']:
            if p is None: 
                continue
            total += p.numel()
            print("   ", tuple(p.shape))
    print("總參數量(元素個數):", total)
def unfreeze_all(policy: nn.Module):
    """
    解凍模型中的所有參數，讓它們全部變為可訓練。

    注意：此函式無法移除已經註冊的梯度掛鉤。
    最穩健的做法是在呼叫此函式後，重新創建一個新的優化器。
    """
    print("\n--- 正在解凍所有模型參數 ---")
    for param in policy.parameters():
        param.requires_grad = True
    print("所有參數已解凍。")        
def run_episode(model, env: gym.Env, seed: int,save_video: bool = False,
    video_dir: str = "./videos",
    video_prefix: str = "episode",
    difficulty: float = None,) -> float:
    """
    在指定的環境和種子下，運行一個完整的 episode 並返回總獎勵。
    """
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        prefix = f"{video_prefix}_seed{seed}"
        if difficulty is not None:
            prefix += f"_diff{difficulty:.2f}"
        recorder = RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix=prefix,
            episode_trigger=lambda ep: True,  # 每回合都錄
        )
        env = recorder
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
        obs = env.reset()
        
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    try:
        env.close()
        if recorder is not None:
            recorder.close()
    except Exception as e:
        print(f"⚠️ 關閉錄影環境時出錯: {e}")
    return total_reward

def compare_models():
    """
    主函式，用於載入並比較多個模型的性能，測試案例隨機抽樣自 hard_seeds.json。
    dad / mom 從 --dad / --mom CLI 傳入；child 從 --ties_out 或預設路徑取得。
    """
    # --- 1. 設定您要比較的模型（從 CLI args 取路徑）---
    def _ext(path):
        return "pkl" if path.endswith(".pkl") else "zip"

    child_path = args.ties_out or f"./models/ties_test_child_{args.crossover}.pkl"
    models_to_test = {}
    if args.dad:
        models_to_test["Dad"] = {"path": args.dad, "type": _ext(args.dad)}
    if args.mom:
        models_to_test["Mom"] = {"path": args.mom, "type": _ext(args.mom)}
    models_to_test[f"Child ({args.crossover})"] = {"path": child_path, "type": _ext(child_path)}

    # --- 2. 設定測試環境與測試案例來源 ---
    env_id = "BipedalWalkerCustom-v0"
    hard_seeds_path = "./logs/hard_seeds.json"
    num_test_cases = args.ties_n_eval  # 用 --ties_n_eval 控制局數（預設 5，可傳 100）
    diff_min_filter = 0.0
    diff_max_filter = 1.0

    print("="*50)
    print(f"正在開始模型性能比較...")
    print(f"將從 '{hard_seeds_path}' 隨機抽樣 {num_test_cases} 個測試案例。")
    print(f"難度範圍: [{diff_min_filter}, {diff_max_filter}]")
    print("="*50)

    # --- 3. 讀取並準備測試案例 (整合 sample_hardseeds 邏輯) ---
    if not os.path.exists(hard_seeds_path):
        print(f"❌ 錯誤: 找不到測試案例檔案 -> {hard_seeds_path}")
        return
        
    try:
        with open(hard_seeds_path, "r") as f:
            hardseed_dict = json.load(f)

        all_pairs = []
        for diff_str, seeds in hardseed_dict.items():
            d = float(diff_str)
            if diff_min_filter <= d <= diff_max_filter:
                for seed_str in seeds.keys():
                    all_pairs.append((d, int(seed_str)))

        # 確保抽樣數量不超過可用總數
        num_to_sample = min(num_test_cases, len(all_pairs))
        if num_to_sample == 0:
            print("❌ 錯誤: 在指定的難度範圍內找不到任何可用的測試案例。")
            return
            
        test_cases = random.sample(all_pairs, num_to_sample)
        print(f"成功抽樣 {len(test_cases)} 個測試案例。")

    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ 錯誤: 無法解析 '{hard_seeds_path}'。請確認檔案格式是否為巢狀字典。錯誤訊息: {e}")
        return

    # --- 4. 載入所有模型到記憶體 ---
    loaded_models = {}
    print("\n正在預先載入所有模型...")
    temp_env = gym.make(env_id)
    for model_name, model_info in models_to_test.items():
        model_path = model_info["path"]
        model_type = model_info["type"]
        
        if not os.path.exists(model_path):
            print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
            continue

        try:
            if model_path.endswith(".pkl"):
                with open(model_path, "rb") as f:
                    policy = cloudpickle.load(f)
                model = PPO("MlpPolicy", temp_env, verbose=0)
                model.policy = policy.to(model.device)
            else:
                load_path = model_path[:-4] if model_path.endswith(".zip") else model_path
                model = PPO.load(load_path, env=temp_env, device="cpu")
            loaded_models[model_name] = model
            print(f"  - ✅ 已載入: {model_name}")
        except Exception as e:
            print(f"  - ❌ 載入失敗 {model_name}: {e}")
    temp_env.close()

    if not loaded_models:
        print("\n❌ 錯誤: 沒有成功載入任何模型，測試中止。")
        return
    os.makedirs("./results", exist_ok=True)
    csv_path = "./results/model_comparison_results.csv"
    # --- 5. 逐一執行測試案例 ---
    all_results = {model_name: [] for model_name in loaded_models.keys()}
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["case_idx", "difficulty", "seed", "model_name", "reward"])
        for i, (difficulty, seed) in enumerate(test_cases):
            print(f"\n--- 測試案例 {i+1}/{len(test_cases)} (Difficulty: {difficulty}, Seed: {seed}) ---")
            
            # env = gym.make(env_id, difficulty=difficulty,render_mode="rgb_array")
            env = gym.make(env_id, difficulty=difficulty)
            for model_name, model in loaded_models.items():
                model.set_env(env)
                reward = run_episode(model, env, seed,save_video=False, video_dir = "./videos",video_prefix= model_name,difficulty=1.0)
                print(f"  - 模型 '{model_name}' 的獎勵: {reward:.2f}")
                all_results[model_name].append(reward)
                writer.writerow([i, difficulty, seed, model_name, reward])
                fcsv.flush()
            env.close()

    # --- 6. 顯示最終的比較報告 ---
    print("\n\n" + "="*50)
    print("📊 最終性能比較報告")
    print("="*50)
    for model_name, rewards in all_results.items():
        if not rewards: continue
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"模型: {model_name}")
        print(f"  - 平均獎勵: {avg_reward:.2f} ± {std_reward:.2f} (基於 {len(rewards)} 次測試)")
        print(f"  - 詳細分數: {[f'{r:.2f}' for r in rewards]}")
        print("-" * 20)
    stats = []
    for name, rewards in all_results.items():
        if rewards:
            stats.append((name, float(np.mean(rewards)), float(np.std(rewards)), len(rewards)))

    if not stats:
        print("⚠️ 無可視化資料（all_results 為空），跳過繪圖。")
        return

    # 依平均獎勵由高到低排序
    stats.sort(key=lambda x: x[1], reverse=True)
    labels = [s[0] for s in stats]
    means = [s[1] for s in stats]
    stds  = [s[2] for s in stats]
    ns    = [s[3] for s in stats]

    # 繪圖
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5)

    ax.set_title("Model Comparison — Average Reward (±1 SD)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 在每根柱上方標註 N（測試次數）
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m, f"n={n}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_path = "./results/model_comparison_bar.png"
    plt.savefig(fig_path, dpi=150)
    print(f"📈 已輸出長條圖：{fig_path}")
def objective(trial: optuna.Trial) -> float:
    """
    Optuna 的目標函式。每一次呼叫，都代表一次完整的端到端實驗。
    """
    print(f"\n\n===== 開始 Optuna Trial #{trial.number} =====")
    
    # --- A. 讓 Optuna 為我們「建議」超參數 ---
    # 這裡只選擇了幾個最重要的作為範例，您可以增加更多
    
    # 離線蒸餾相關
    offline_lr = trial.suggest_float("offline_lr", 1e-6, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 1.0)
    temperature = trial.suggest_float("temperature", 1.0, 10.0)
    offline_epochs = trial.suggest_int("offline_epochs", 1, 20)

    # 線上微調相關
    online_lr = trial.suggest_float("online_lr", 1e-7, 1e-3, log=True)
    pure_lr_scale_factor = trial.suggest_float("pure_lr_scale_factor", 0.001, 0.5, log=True)
    finetune_steps = 100_000 # 為了快速迭代，可以先用較少的步數

    print("本次 Trial 的超參數:")
    print(trial.params)

    # --- B. 執行您完整的融合與訓練流程 ---
    try:
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval1/fitness_top.json"
        distill_pt   = "./logs/ga_eval/mtkd_continuous.pt(2)"
        env_id       = "BipedalWalkerCustom-v0"

        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        
        # mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model4", "model5"
        
        # resolve_path 函式保持不變
        def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
             # 應該加上找不到的處理
            raise FileNotFoundError(f"在 '{root}' 資料夾中找不到模型檔案 '{name}' (或 .zip)")

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f"Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)

        mom = PPO.load(mom_path, env=env)
        dad = PPO.load(dad_path, env=env)
        # 2.1 建構全新的「雙子網路」策略
        child_policy = create_dual_channel_policy(dad.policy, mom.policy)
        # 2.2 交叉通道初始化
        if args.crossover == "ties":
            print("[crossover=ties] 用 TIES 初始化交叉通道")
            ties_initialize_crosstalk(child_policy, dad.policy, mom.policy, k=args.ties_k)
        elif args.crossover == "ot":
            print("[crossover=ot] 用 OT Fusion 初始化交叉通道")
            ot_initialize_crosstalk(child_policy, dad.policy, mom.policy)
        elif args.crossover == "ot_recursive":
            print("[crossover=ot_recursive] 用遞迴 OT 初始化交叉通道")
            recursive_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, env=env)
        elif args.crossover == "ot_iter":
            print(f"[crossover=ot_iter] 用迭代 OT 初始化交叉通道（{args.ot_iters} 次）")
            iterative_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=args.ot_iters)
        elif args.crossover == "ot_mutual":
            print(f"[crossover=ot_mutual] 兩通道互相 OT 融合（{args.ot_iters} 輪）")
            mutual_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=max(args.ot_iters, 100), tol=args.ot_tol, obs_batch=_sample_obs_batch(env, n=256))
        else:
            print("[crossover=zero] 用小噪音初始化交叉通道")
            zero_initialize_crosstalk(child_policy)
        # 2.3 創建一個新的 PPO Agent 來承載我們的子代策略
        # 注意：這裡的超參數應該與您的 dad/mom agent 保持一致

        child_agent = PPO(
            policy=dad.policy.__class__,
            env=env,
            learning_rate=online_lr, # 線上微調時的學習率
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
            verbose=0
        )
        child_agent.policy = child_policy.to(child_agent.device)
        
        # 2.4 【關鍵】為子代策略設定我們特製的「精細版差異化優化器」
        # 注意：線上微調時，我們可能希望交叉通道的學習率高一些，所以這裡用了 lr_pure/10
        child_agent.policy.optimizer = create_refined_differential_optimizer(
            policy=child_agent.policy,
            lr_pure=child_agent.learning_rate, # 使用 PPO agent 的主學習率
            cross_lr_scale_factor=pure_lr_scale_factor # 線上微調時，給予交叉通道 10% 的學習率
        )
        
        # 2.5 執行「全局離線蒸餾」作為熱身 (使用軟硬結合的損失函數)
        print("\n--- 開始執行離線蒸餾預訓練 ---")
        train_network_offline(
            ppo_model=child_agent,
            pt_path=distill_pt,
            epochs=offline_epochs,             # 離線訓練10輪
            lr=offline_lr,               # 離線訓練使用稍大的學習率
            vf_coef=0.5,
            temperature=temperature,       # 軟目標的溫度
            alpha=alpha,             # 軟目標的權重
            device=child_agent.device
        )
        
        # 2.6 執行最終的「線上微調」
        print("\n--- 開始執行線上強化學習微調 ---")
        finetune_steps = 100000
        auto_difficulty_callback = AutoDifficultyCallback(
        env,None, eval_freq=10000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=False,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
        child_agent.learn(total_timesteps=finetune_steps, callback=[ auto_difficulty_callback,],progress_bar=True)
        # --- C. 評估最終性能並返回分數 ---
        # 為了節省時間，可以用較少的 episode 進行評估
        mean_reward, std_reward = evaluate_policy(child_agent, env, n_eval_episodes=20)
        final_difficulty = env.unwrapped.difficulty
        print(f"Trial #{trial.number} 完成，平均獎勵: {mean_reward}")
        difficulty_bonus = 150.0
        composite_score = mean_reward + 250*((final_difficulty-0.5)/0.05)
        env.close()
        return composite_score

    except Exception as e:
        print(f"Trial #{trial.number} 因錯誤而失敗: {e}")
        # 告知 Optuna 這次嘗試失敗了
        print("--- DETAILED TRACEBACK ---")
        traceback.print_exc()
        print("---------")
        return -1000.0 # 返回一個極差的分數
def analyze_heatmaps(policies: dict, save_dir: str):
    """
    為多個模型的每一層權重繪製熱力圖，並並排儲存以便比較。
    """
    print("\n--- 正在生成權重熱力圖 ---")
    # 獲取所有模型共享的層名稱
    sample_policy = next(iter(policies.values()))
    layer_names = [name for name, module in sample_policy.named_modules() if isinstance(module, nn.Linear)]

    with torch.no_grad():
        for name in layer_names:
            num_models = len(policies)
            fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
            if num_models == 1: axes = [axes] # 處理只有一個模型的情況

            for i, (model_name, policy) in enumerate(policies.items()):
                module = dict(policy.named_modules())[name]
                W = module.weight.data.cpu().numpy()
                ax = axes[i]
                
                sns.heatmap(W, cmap='coolwarm', center=0.0, ax=ax, cbar=i==num_models-1)
                ax.set_title(f"{model_name}\nShape: {W.shape}", fontsize=14)
                
                # 如果是子代模型，畫出輔助線
                if "child" in model_name.lower():
                    out_half, in_half = W.shape[0] // 2, W.shape[1] // 2
                    if out_half > 0: ax.axhline(out_half, color='black', linewidth=2.5)
                    if in_half > 0: ax.axvline(in_half, color='black', linewidth=2.5)

            fig.suptitle(f"Weighted Heatmap Comparison: Layer '{name}'", fontsize=20, fontweight='bold')
            plt.rcParams['axes.unicode_minus'] = False
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            save_path = os.path.join(save_dir, f"heatmap_{name.replace('.', '_')}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  - 已儲存 '{name}' 層的熱力圖比較至: {save_path}")
def analyze_weight_difference(child_policy: nn.Module, dad_policy: nn.Module, mom_policy: nn.Module, save_dir: str):
    """
    計算並視覺化子代模型權重與其親代對應部分之間的差異。
    上半部分與父親比較，下半部分與母親比較。
    """
    print("\n--- 正在生成權重【差異】熱力圖 ---")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for name, child_module in child_policy.named_modules():
            if isinstance(child_module, nn.Linear):
                dad_module = dict(dad_policy.named_modules()).get(name)
                mom_module = dict(mom_policy.named_modules()).get(name)
                
                # 確保父母雙方都有對應的層
                if not (dad_module and mom_module): 
                    print(f"  - 跳過層 '{name}': 在父代或母代中找不到對應層。")
                    continue
                
                Wc = child_module.weight.data
                Wd = dad_module.weight.data
                Wm = mom_module.weight.data

                # 檢查維度是否匹配，以及是否可分割
                if Wc.shape != Wd.shape or Wc.shape != Wm.shape:
                    print(f"  - 跳過層 '{name}': 維度不匹配。")
                    continue
                
                out_half = Wc.shape[0] // 2
                if out_half == 0 or Wc.shape[0] % 2 != 0:
                    print(f"  - 跳過層 '{name}': 輸出維度無法被對半分。")
                    continue

                # --- 核心計算邏輯 ---
                # 計算上半部分的差異 (Child - Dad)
                diff_top = Wc[:out_half, :] - Wd[:out_half, :]
                
                # 計算下半部分的差異 (Child - Mom)
                diff_bottom = Wc[out_half:, :] - Wm[out_half:, :]
                
                # 將兩個差異區塊拼接回來，形成一個完整的差異矩陣
                diff_matrix = torch.cat([diff_top, diff_bottom], dim=0).cpu().numpy()
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                # --- 繪圖 ---
                plt.figure(figsize=(10, 8))
                
                # 找到差異的最大絕對值，用於設定對稱的顏色標尺
                vmax = np.abs(diff_matrix).max()
                
                # 使用熱力圖視覺化差異矩陣
                sns.heatmap(diff_matrix, cmap='coolwarm', center=0.0, vmin=-vmax, vmax=vmax)
                
                # 繪製輔助線，標示出不同的比較區域
                ax = plt.gca()
                ax.axhline(out_half, color='black', linewidth=2.5)

                ax.set_title(f"權重差異圖 (訓練後變化): Layer '{name}'\n上半部: Child - Dad | 下半部: Child - Mom", fontsize=16)
                plt.xlabel("輸入特徵 (Input Features)")
                plt.ylabel("輸出神經元 (Output Neurons)")
                plt.tight_layout()
                
                # --- 儲存圖表 ---
                save_path = os.path.join(save_dir, f"difference_heatmap_{name.replace('.', '_')}.png")
                plt.savefig(save_path, dpi=150)
                plt.close() # 關閉圖表，釋放記憶體
                print(f"  - 已儲存 '{name}' 層的差異圖至: {save_path}")
def analyze_histograms(policy: nn.Module, save_dir: str, model_name: str):
    """
    為指定模型的每一層權重，繪製四個通道的權重分佈直方圖。
    """
    print(f"\n--- 正在為 '{model_name}' 生成權重分佈直方圖 ---")
    with torch.no_grad():
        for name, module in policy.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                out_half, in_half = W.shape[0] // 2, W.shape[1] // 2
                
                if in_half == 0 or out_half == 0: continue # 跳過無法分割的層

                blocks = {
                    "父->父 (左上)": W[:out_half, :in_half].flatten().cpu().numpy(),
                    "母->父 (右上)": W[:out_half, in_half:].flatten().cpu().numpy(),
                    "父->母 (左下)": W[out_half:, :in_half].flatten().cpu().numpy(),
                    "母->母 (右下)": W[out_half:, in_half:].flatten().cpu().numpy(),
                }
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f"'{model_name}' - Layer '{name}' 權重分佈", fontsize=20, fontweight='bold')
                
                for ax, (title, data) in zip(axes.flatten(), blocks.items()):
                    sns.histplot(data, bins=50, kde=True, ax=ax)
                    ax.set_title(title)
                    ax.set_xlabel("權重值")
                    ax.set_ylabel("數量")
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                save_path = os.path.join(save_dir, f"histogram_{name.replace('.', '_')}.png")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"  - 已儲存 '{name}' 層的直方圖至: {save_path}")
def collect_actions(model: PPO, env: gym.Env, seed: int) -> np.ndarray:
    """在指定的環境和種子碼下，收集模型完整的動作序列。"""
    actions = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        # 使用確定性動作，以確保策略是固定的
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(actions)
def collect_trajectories(model: PPO, env: gym.Env, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """在指定的環境和種子碼下，收集模型完整的【狀態】和【動作】序列。"""
    observations = []
    actions = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        observations.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(observations), np.array(actions)
def plot_action_distributions(actions_dict: dict, save_dir: str, difficulty: float, seed: int):
    """
    為多個模型的動作序列，繪製每個動作維度的分佈直方圖/密度圖。
    """
    print(f"\n✅ 正在繪製動作分佈直方圖...")
    
    # 獲取動作維度
    sample_actions = next(iter(actions_dict.values()))
    action_dim = sample_actions.shape[1]

    for i in range(action_dim):
        plt.figure(figsize=(12, 8))
        
        # 為每個模型繪製核密度估計圖 (KDE Plot)，更平滑
        for model_name, actions in actions_dict.items():
            sns.kdeplot(actions[:, i], label=model_name, fill=True, alpha=0.5)

        plt.title(f'動作維度 {i+1} 的分佈比較\n(Difficulty: {difficulty}, Seed: {seed})', fontsize=16)
        plt.xlabel(f'動作 {i+1} 的數值', fontsize=12)
        plt.ylabel('密度 (Density)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"action_distribution_dim_{i+1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   - 已儲存動作維度 {i+1} 的分佈圖至: {save_path}")
def run_state_conditioned_analysis():
    """
    執行完整的、基於物理特徵的狀態條件下的動作分析流程。
    """
    print("="*60)
    print("      模型在特定物理狀態下的動作對比分析")
    print("="*60)
    env_id = "BipedalWalkerCustom-v0"
    best_difficulty=0.8
    best_seed=3729259198
    conditions = {
        "高速前進時 (vel_x > 1.0)": 
            lambda obs: obs[2] > 1.0,
        "身體失衡後仰時 (hull_ang_vel < -0.3)": 
            lambda obs: obs[1] < -0.3,
        "處於滯空/下墜狀態時 (vel_y < -0.5)":
            lambda obs: obs[3] < -0.5,
        "左腿大步邁出時 (hip_joint_1_angle < -0.3)":
            lambda obs: obs[4] < -0.3,
        "即將撞上前方障礙物時 (lidar[0] < 0.5)":
            lambda obs: obs[14] < 0.5, # lidar[0] 是正前方的雷達
    }
    print("已定義的關鍵物理情境:")
    for name in conditions.keys():
        print(f"  - {name}")

    # --- 2. 載入模型並收集狀態-動作軌跡 ---
    env = gym.make(env_id, difficulty=best_difficulty)
    
    

    print(f"\n--- 正在收集軌跡數據 (Difficulty: {best_difficulty}, Seed: {best_seed}) ---")
    models_to_test = {
        "Child_model (Evolved Child)": {
            "path": "./models/child_model2_x_model3_fused_full(5).pkl",
            "type": "pkl"
        },
        "Dad_model (Original Parent-Dad)": {
            "path": "./best_model/model2.zip",
            "type": "zip"
        
        },
        
        "Mom_model (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model3.zip",
        "type": "zip"
        },

        

        
    }
    
    
    # --- 2. 設定測試環境與測試案例來源 ---
    save_dir = "./actions_stept"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    loaded_models = {}
    print("\n正在預先載入所有模型...")
    temp_env = gym.make(env_id)
    for model_name, model_info in models_to_test.items():
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
                continue

            model = None
            if model_type == "zip":
                model = PPO.load(model_path, env=temp_env)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    model = cloudpickle.load(f)
            
            if model:
                loaded_models[model_name] = model
                print(f"  - ✅ 已載入: {model_name}")
    temp_env.close()
    trajectories = {}
    for name, model in loaded_models.items():
        trajectories[name] = collect_trajectories(model, env, best_seed)
    
    min_len = min(len(obs) for obs, act in trajectories.values())
    
    # --- 3. 逐一分析每個情境 ---
    os.makedirs(save_dir, exist_ok=True)
    
    for cond_name, cond_func in conditions.items():
        print(f"\n--- 正在分析情境: {cond_name} ---")
        
        child_obs, _ = trajectories["Child_model (Evolved Child)"]
        condition_indices = [i for i, obs in enumerate(child_obs[:min_len]) if cond_func(obs)]

        if not condition_indices:
            print("  -> 在此回合中，子代模型未觸發此情境。")
            continue
            
        print(f"  -> 在 {min_len} 幀中，共找到 {len(condition_indices)} 個符合條件的時間點。")

        action_data = []
        for model_name, (obs_seq, act_seq) in trajectories.items():
            for idx in condition_indices:
                action = act_seq[idx]
                for dim in range(action.shape[0]):
                    action_data.append({
                        "模型": model_name,
                        "動作維度": f"動作 {dim+1}",
                        "動作值": action[dim]
                    })
        
        df_actions = pd.DataFrame(action_data)

        # --- 4. 繪製並儲存該情境下的動作比較圖 ---
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(figsize=(15, 8))
        
        sns.boxplot(data=df_actions, x="動作維度", y="動作值", hue="模型", palette="pastel")
        sns.stripplot(data=df_actions, x="動作維度", y="動作值", hue="模型", dodge=True, jitter=0.2, palette="deep", size=4)

        plt.title(f'關鍵情境下的動作比較: {cond_name}', fontsize=18)
        plt.ylabel("動作值 (Action Value)", fontsize=12)
        plt.xlabel("動作維度", fontsize=12)
        plt.legend(title='模型')
        plt.grid(True, linestyle='--')
        plt.tight_layout()

        safe_cond_name = "".join(c for c in cond_name if c.isalnum() or c in (' ', '_')).rstrip()
        save_path = os.path.join(save_dir, f"state_conditioned_{safe_cond_name.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> ✅ 已儲存分析圖表至: {save_path}")

    env.close()
    print("\n\n分析完成！")
def run_action_analysis():
    """
    執行完整的動作相似度分析流程。
    """
    print("="*60)
    print("                模型動作相似度分析")
    print("="*60)

    env_id = "BipedalWalkerCustom-v0"
    # --- 2. 載入模型並收集動作序列 ---
    env = gym.make(env_id, difficulty=0.70)
    best_seed=587663355
    models_to_test = {
        "Child_model (Evolved Child)": {
            "path": "./models/child_model5_x_model4_fused_full.pkl",
            "type": "pkl"
        },
        "Dad_model (Original Parent-Dad)": {
            "path": "./best_model/model4.zip",
            "type": "zip"
        
        },
        
        "Mom_model (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model5.zip",
        "type": "zip"
        },

        

        
    }
    best_difficulty=0.85
    # --- 2. 設定測試環境與測試案例來源 ---
    save_dir = "./actions_analysis1"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    loaded_models = {}
    print("\n正在預先載入所有模型...")
    temp_env = gym.make(env_id)
    for model_name, model_info in models_to_test.items():
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
                continue

            model = None
            if model_type == "zip":
                model = PPO.load(model_path, env=temp_env)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    model = cloudpickle.load(f)
            
            if model:
                loaded_models[model_name] = model
                print(f"  - ✅ 已載入: {model_name}")
    temp_env.close()
    loaded_policies = {name: agent.policy for name, agent in loaded_models.items()}
    child_policy = loaded_policies.get("Child_model (Evolved Child)")
    mom_policy = loaded_policies.get( "Mom_model (Original Parent - Mom)")
    dad_policy = loaded_policies.get( "Dad_model (Original Parent-Dad)")
    print("\n--- 正在收集動作序列... ---")
    actions_dad = collect_actions(dad_policy, env, best_seed)
    actions_mom = collect_actions(mom_policy, env, best_seed)
    actions_child = collect_actions(child_policy, env, best_seed)
    
    # 確保長度一致以便比較
    min_len = min(len(actions_dad), len(actions_mom), len(actions_child))
    actions_dad = actions_dad[:min_len]
    actions_mom = actions_mom[:min_len]
    actions_child = actions_child[:min_len]
    print(f"動作序列收集完成，已對齊長度至 {min_len} 幀。")

    # --- 3. 計算並報告動作相似度 ---
    # 使用 L2 距離 (歐氏距離的平方) 的平均值，即 MSE
    mse_child_dad = np.mean((actions_child - actions_dad)**2)
    mse_child_mom = np.mean((actions_child - actions_mom)**2)

    print("\n\n" + "="*60)
    print("📊 動作相似度分析報告 (均方誤差 MSE，越低越相似)")
    print("="*60)
    print(f"  - 子代 vs. 父代 (Child vs. Dad): {mse_child_dad:.6f}")
    print(f"  - 子代 vs. 母代 (Child vs. Mom): {mse_child_mom:.6f}")
    
    if mse_child_dad < mse_child_mom:
        print("\n【結論】: 在此案例中，子代的行為模式更接近【父代】。")
    elif mse_child_mom < mse_child_dad:
        print("\n【結論】: 在此案例中，子代的行為模式更接近【母代】。")
    else:
        print("\n【結論】: 在此案例中，子代與父母的行為相似度幾乎相同。")


    # --- 4. 視覺化分析 ---
    os.makedirs(save_dir, exist_ok=True)
    
    # 4a. 繪製總體相似度長條圖
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['vs. 父代 (Dad)', 'vs. 母代 (Mom)'], y=[mse_child_dad, mse_child_mom], palette='pastel')
    plt.title('子代與父母的動作策略相似度', fontsize=16)
    plt.ylabel('平均均方誤差 (MSE)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "action_similarity_mse.png"), dpi=300)
    print(f"\n✅ 已儲存總體相似度長條圖至 '{save_dir}/action_similarity_mse.png'")

    # 4b. 繪製逐幀動作對比曲線圖
    action_dim = actions_child.shape[1]
    timesteps = np.arange(min_len)
    
    fig, axes = plt.subplots(action_dim, 1, figsize=(18, 5 * action_dim), sharex=True)
    if action_dim == 1: axes = [axes]
        
    fig.suptitle(f'逐幀動作對比 (Difficulty: {best_difficulty}, Seed: {best_seed})', fontsize=20, y=0.97)

    for i in range(action_dim):
        axes[i].plot(timesteps, actions_dad[:, i], label='父代 (Dad)', color='royalblue', alpha=0.7, linewidth=1.5)
        axes[i].plot(timesteps, actions_mom[:, i], label='母代 (Mom)', color='crimson', alpha=0.7, linewidth=1.5)
        axes[i].plot(timesteps, actions_child[:, i], label='子代 (Child)', color='forestgreen', linewidth=2.5)
        axes[i].set_ylabel(f'動作維度 {i+1}')
        axes[i].legend()
        axes[i].grid(True, linestyle='--')

    axes[-1].set_xlabel('時間步 (Timestep)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "action_trajectory_comparison.png"), dpi=300)
    print(f"✅ 已儲存逐幀動作對比曲線圖至 '{save_dir}/action_trajectory_comparison.png'")
    actions_dict = {
        "父代 (Dad)": actions_dad,
        "母代 (Mom)": actions_mom,
        "子代 (Child)": actions_child
    }
    plot_action_distributions(actions_dict, save_dir, best_difficulty, best_seed)
    
    env.close()
    print("\n分析完成！所有圖表已儲存至:", save_dir)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test",  action="store_true")
    parser.add_argument("--prune-hardseeds", action="store_true")
    parser.add_argument("--rename_modelname", action="store_true")
    parser.add_argument("--test_population", action="store_true")
    parser.add_argument("--evolved", action="store_true")
    parser.add_argument("--evolved_all", action="store_true")
    parser.add_argument("--compare_test", action="store_true")
    parser.add_argument("--find_value", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--weight", action="store_true")
    parser.add_argument("--action", action="store_true")
    parser.add_argument("--action_state", action="store_true")
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--test_all_generations", action="store_true")
    parser.add_argument("--test_ties", action="store_true")
    parser.add_argument("--dad",  type=str, default=None, help="dad 模型路徑 (.zip/.pkl)")
    parser.add_argument("--mom",  type=str, default=None, help="mom 模型路徑 (.zip/.pkl)")
    parser.add_argument("--ties_k", type=float, default=0.2, help="TIES top-k 比例（預設 0.2）")
    parser.add_argument("--ties_sign", type=str, default="mass", choices=["mass","normfrac","normmass"])
    parser.add_argument("--ties_n_eval", type=int, default=5, help="測試局數")
    parser.add_argument("--ties_out", type=str, default=None,
                         help="融合後 child 存檔路徑，預設 None 時會自動帶入 --crossover 的值"
                              "（./models/ties_test_child_<crossover>.pkl），避免不同 crossover 變體互相覆蓋")
    parser.add_argument("--distill_pt", type=str, default="./logs/ga_eval/mtkd_continuous.pt(2)", help="蒸餾資料 .pt 路徑")
    parser.add_argument("--progressive_steps_per_stage", type=int, default=20_000,
                         help="ot_progressive / ties_progressive 每層退火訓練的步數")
    parser.add_argument("--progressive_max_stages", type=int, default=None,
                         help="ot_progressive 只跑前幾層就停（預設 None＝全部層都跑），用來測試只換單一層的效果")
    parser.add_argument("--progressive_no_anneal", action="store_true",
                         help="ot_progressive 不做漸進退火，每個 stage 一開始就直接是完整 OT 值，用來對照有無退火的差異")
    parser.add_argument("--progressive_n_rounds", type=int, default=10,
                         help="ot_progressive_iter 每一層內層迴圈跑幾輪 OT→訓練（預設 10）")
    parser.add_argument("--progressive_steps_per_round", type=int, default=1_000_000,
                         help="ot_progressive_iter 全部層蒸餾消化完之後，最終真實環境微調的步數（預設 100 萬步）")
    parser.add_argument("--progressive_pre_finetune_out", type=str, default=None,
                         help="ot_progressive_iter(_focused) 在最終 PPO 微調『開始之前』額外存一份純 OT+蒸餾初始化模型的路徑；不給則自動用 <ties_out 去掉副檔名>_preppo.pkl。設成空字串可停用")
    parser.add_argument("--progressive_ot_alpha_init", type=float, default=1.0,
                         help="ot_progressive_iter(_focused) 每個 stage 第一輪(round_idx=0)的 OT 介入強度 alpha（預設 1.0＝第一輪完整套用 OT，大力對齊）")
    parser.add_argument("--progressive_ot_gamma", type=float, default=0.7169,
                         help="ot_progressive_iter(_focused) alpha 隨 round 的指數衰減率：alpha_t = alpha_init * gamma**round_idx（預設 0.7169，搭配 progressive_n_rounds=10 時最後一輪 alpha≈0.05，幾乎退化到跟 distill_only 一樣；若改動 n_rounds 需重新用 gamma=(alpha_target/alpha_init)**(1/(n_rounds-1)) 反推）")
    parser.add_argument("--ot_frac", type=float, default=1.0,
                         help="OT 只覆寫交叉通道前多少比例的列（輸出神經元），其餘維持蒸餾基底/靜音；1.0=整塊覆寫（原本行為），適用 ot_progressive / ot_first_distill_rest")
    parser.add_argument("--crossover", type=str, default="zero", choices=["zero", "ties", "ties_progressive", "ot", "ot_recursive", "ot_progressive", "ot_progressive_iter", "ot_progressive_iter_focused", "ot_reconstruct", "ot_iter", "ot_mutual", "ot_first_distill_rest", "distill_only"],
                        help="交叉通道初始化方式：zero=小噪音（原版）, ties=TIES融合, ot=OT Fusion, ot_recursive=遞迴OT, ot_progressive_iter=漸進式逐層+逐輪迭代OT（每層固定輪數，OT↔蒸餾交替，每輪蒸餾全部交叉通道）, ot_progressive_iter_focused=跟 ot_progressive_iter 相同但每輪蒸餾只聚焦剛 OT 對齊過的那一層，其餘層凍結，開跑前先蒸餾全部交叉通道一次當基底, ot_iter=迭代OT（多次精化）, ot_mutual=兩通道互相OT融合收斂, ot_first_distill_rest=只對第一層做OT並凍結，其餘層改用蒸餾適應, distill_only=完全不做OT，全部交叉通道只用蒸餾（對照組）")
    parser.add_argument("--ot_iters", type=int, default=5,
                        help="ot_iter 迭代次數（預設 5）；ot_mutual 的最大輪數（預設 20）")
    parser.add_argument("--ot_tol", type=float, default=1e-3,
                        help="ot_mutual 收斂門檻：|X_md - X_dm| 低於此值時提早停止（預設 1e-3）")
    args = parser.parse_args()

    if args.train:
        more_train()
    elif args.test:
        test()
    elif args.compare_test:
        compare_models()
    elif args.prune_hardseeds:
        prune_hardseeds()
    elif args.rename_modelname:
        rename_modelname()
    elif args.test_population:
        model_folder = "./test_models"
        model_files = sorted([
        os.path.join(model_folder, f)
        for f in os.listdir(model_folder)
        if f.endswith(".zip") or f.endswith(".pkl")
    ])
        # 評估並儲存資料
        fitness_out, top_names = test_population(
        model_paths=model_files,
        hardseed_path="./logs/hard_seeds.json",
        env_id="BipedalWalkerCustom-v0",
        num_samples=1000,
        eval_n_stage1=1,
        eval_n_stage2=10,
        reward_threshold=250,
        top_k=10,                      # 你要的 Top-30
        save_dir="./logs/ga_eval1",
        save_name="mtkd_continuous.pt(1)",
        num_workers=10,
        device_str="cpu",              # 想用 GPU 改成 "cuda"
        diff_min=0.0,
        diff_max=1.0,
        generate_distill=False,
    )
    elif args.action:
        run_action_analysis()
    elif args.action_state:
        run_state_conditioned_analysis()
    elif args.weight:
        models_to_test = {
        "Child_model (Evolved Child)": {
            "path": "./models/child_model2_x_model3_fused_full(5).pkl",
            "type": "pkl"
        },
        "Dad_model (Original Parent-Dad)": {
            "path": "./best_model/model2.zip",
            "type": "zip"
        
        },
        
        "Mom_model (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model3.zip",
        "type": "zip"
        },

        

        
    }

    # --- 2. 設定測試環境與測試案例來源 ---
        env_id = "BipedalWalkerCustom-v0"
        save_dir = "./weight_analysis"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        loaded_models = {}
        print("\n正在預先載入所有模型...")
        temp_env = gym.make(env_id)
        for model_name, model_info in models_to_test.items():
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
                continue

            model = None
            if model_type == "zip":
                model = PPO.load(model_path, env=temp_env)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    model = cloudpickle.load(f)
            
            if model:
                loaded_models[model_name] = model
                print(f"  - ✅ 已載入: {model_name}")
        temp_env.close()
        loaded_policies = {name: agent.policy for name, agent in loaded_models.items()}
        child_policy = loaded_policies.get("Child_model (Evolved Child)")
        mom_policy = loaded_policies.get( "Mom_model (Original Parent - Mom)")
        dad_policy = loaded_policies.get("Dad_model (Original Parent-Dad)")
        analyze_heatmaps(loaded_policies, save_dir=os.path.join(save_dir, "heatmaps"))
        analyze_weight_difference(child_policy=child_policy,mom_policy=mom_policy,dad_policy=dad_policy,save_dir=save_dir)
    # 方法二：直方圖 (只對冠軍子代做)
        if child_policy:
            analyze_histograms(child_policy, save_dir=os.path.join(save_dir, "histograms"), model_name="Champion_Child")
        if mom_policy:
            analyze_histograms(mom_policy, save_dir=os.path.join(save_dir, "histograms_mom"), model_name="Mom")
        if dad_policy:
            analyze_histograms(dad_policy, save_dir=os.path.join(save_dir, "histograms_dad"), model_name="Dad")
        print("\n\n分析完成！所有圖表已儲存至:", save_dir)
    elif args.offline: 
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval/fitness_top.json"
        distill_pt   = "./logs/ga_eval1/mtkd_continuous.pt(1)"
        env_id       = "BipedalWalkerCustom-v0"
        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        
        # mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model3", "model2"
        
        # resolve_path 函式保持不變
        def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
            return None # 應該加上找不到的處理

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f"Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)

        mom = PPO.load(mom_path, env=env)
        dad = PPO.load(dad_path, env=env)
        child_agent = PPO(
            policy=dad.policy.__class__,
            env=env,
            learning_rate=4.567629084674937e-06, # 線上微調時的學習率
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
            verbose=0
        )
        distill_lr = 1e-4 # 為蒸餾設定一個合理的學習率
        optimizer = torch.optim.Adam(child_agent.policy.parameters(), lr=distill_lr)
        train_network_offline(
            ppo_model=child_agent,
            pt_path=distill_pt,
            optimizer=optimizer,
            epochs=1,             # 離線訓練10輪
            lr= 1e-4,               # 離線訓練使用稍大的學習率
            vf_coef=0.5,
            temperature=4.414784424905544,       # 軟目標的溫度
            alpha=0.5962410908066522,             # 軟目標的權重
            device=child_agent.device
        )
        print("\n--- 開始執行線上強化學習微調 ---")
        finetune_steps = 100000
        auto_difficulty_callback = AutoDifficultyCallback(
        env,None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=False,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
        child_agent.learn(total_timesteps=finetune_steps, callback=[ auto_difficulty_callback,],progress_bar=True)

        evolved = child_agent # 將訓練完成的 agent 賦值給 evolved 變數
        
        # --- 步驟 3: 儲存訓練好的模型 (這部分保持不變) ---
        print("\n--- 正在儲存融合後的模型 ---")
        out_dir = Path("./models"); out_dir.mkdir(parents=True, exist_ok=True)
        dad_base = os.path.splitext(os.path.basename(dad_path))[0]
        mom_base = os.path.splitext(os.path.basename(mom_path))[0]
        tag = f"child_{dad_base}_x_{mom_base}_fused" # 加上 fused 標籤以區分
        zip_path = out_dir / f"{tag}.zip"
        evolved.save(str(zip_path))
        pkl_path = out_dir / f"{tag}_full.pkl"
        evolved.env = None # 移除 env 以便打包
        with open(pkl_path, "wb") as f:
            # 需要 import cloudpickle
            import cloudpickle
            cloudpickle.dump(evolved, f)

        print("✅ Saved:")
        print(f"   - SB3 zip:   {zip_path}")
        print(f"   - Full .pkl: {pkl_path}")    
    elif args.evolved: 
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval/fitness_top.json"
        distill_pt   = "./logs/ga_eval1/mtkd_continuous.pt(1)"
        env_id       = "BipedalWalkerCustom-v0"

        # 讀榜單並抽兩個不同模型名
        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        #mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model4","model5"
        # 把名稱解析成實際檔案路徑
        def resolve_path(name: str, root: str) -> str:
            # 絕對路徑直接用
            if os.path.isabs(name) and os.path.exists(name):
                return name

            candidates = []
            if name.endswith(".zip"):
                candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))

            for p in candidates:
                if os.path.exists(p):
                    return p

            # 最後保險：掃描資料夾，把去副檔名後的名字比對

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f" Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        # 建環境（要傳環境物件給 PPO，不是字串）
        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.0)

        # 載入父母
        mom = PPO.load(mom_path, env=env)  # 可加 device="cpu"/"cuda"
        dad = PPO.load(dad_path, env=env)
        debug_plan = {
        1: ("mlp_extractor.policy_net", 0),                # 第 1 輪：強制交換整個 action_net
      # 第 2 輪：強制交換 value 主幹的第 2 層
        2: ("action_net", -1), 
        }
        print("  DEBUG MODE: Running with a predefined evolution plan:")
        for round_num, action in debug_plan.items():
            print(f"  - Round {round_num}: Crossover unit -> {action[0]} (idx={action[1]})")
        # 交配 + progressive
        evolved = progressive_evolve(
            dad, mom, env, distill_pt=distill_pt,debug_plan=debug_plan
        )

        # 輸出檔名：child_父_x_母.zip
        out_dir = Path("./models"); out_dir.mkdir(parents=True, exist_ok=True)
        dad_base = os.path.splitext(os.path.basename(dad_path))[0]
        mom_base = os.path.splitext(os.path.basename(mom_path))[0]
        tag = f"child_{dad_base}_x_{mom_base}"

        # A) SB3 官方格式（最穩，推薦）
        zip_path = out_dir / f"{tag}.zip"
        evolved.save(str(zip_path))

        # B) 輕量化：只存 policy 權重（GA/蒸餾方便）

        # C) 最後手段：完整 pickle（移除 env，避免把環境也打包）
        pkl_path = out_dir / f"{tag}_full.pkl"
        evolved.env = None
        with open(pkl_path, "wb") as f:
            cloudpickle.dump(evolved, f)

        print("✅ Saved:")
        print(f"   - SB3 zip:      {zip_path}")
        print(f"   - Full .pkl:    {pkl_path}")
    elif args.evolved_all:
    # --- 步驟 1: 參數設定與父母模型載入 (這部分保持不變) ---
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval/fitness_top.json"
        distill_pt   = "./logs/ga_eval/mtkd_continuous.pt(2)"
        env_id       = "BipedalWalkerCustom-v0"

        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        
        mom_name, dad_name = random.sample(names, 2)
        # mom_name, dad_name = "model2", "model3"
        
        # resolve_path 函式保持不變
        def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
            return None # 應該加上找不到的處理

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f"Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)

        mom = PPO.load(mom_path, env=env)
        dad = PPO.load(dad_path, env=env)
        # 2.1 建構全新的「雙子網路」策略
        child_policy = create_dual_channel_policy(dad.policy, mom.policy)
        # 2.2 交叉通道初始化
        if args.crossover == "ties":
            print("[crossover=ties] 用 TIES 初始化交叉通道")
            ties_initialize_crosstalk(child_policy, dad.policy, mom.policy, k=args.ties_k)
        elif args.crossover == "ot":
            print("[crossover=ot] 用 OT Fusion 初始化交叉通道")
            ot_initialize_crosstalk(child_policy, dad.policy, mom.policy)
        elif args.crossover == "ot_recursive":
            print("[crossover=ot_recursive] 用遞迴 OT 初始化交叉通道")
            recursive_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, env=env)
        elif args.crossover == "ot_iter":
            print(f"[crossover=ot_iter] 用迭代 OT 初始化交叉通道（{args.ot_iters} 次）")
            iterative_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=args.ot_iters)
        elif args.crossover == "ot_mutual":
            print(f"[crossover=ot_mutual] 兩通道互相 OT 融合（{args.ot_iters} 輪）")
            mutual_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=max(args.ot_iters, 100), tol=args.ot_tol, obs_batch=_sample_obs_batch(env, n=256))
        else:
            print("[crossover=zero] 用小噪音初始化交叉通道")
            zero_initialize_crosstalk(child_policy)
        # 2.3 創建一個新的 PPO Agent 來承載我們的子代策略
        # 注意：這裡的超參數應該與您的 dad/mom agent 保持一致
        child_agent = PPO(
            policy=dad.policy.__class__,
            env=env,
            learning_rate=4.567629084674937e-06, # 線上微調時的學習率
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
            verbose=0
        )
        child_agent.policy = child_policy.to(child_agent.device)
        trainable_params, hook_handles = freeze_pure_channels(child_agent.policy,bias_mode="freeze",allow_train_if_unsplit=False)
        child_agent.policy.optimizer = torch.optim.Adam(trainable_params, lr=0.008780571292219902,weight_decay=0.0)
        list_optimizer_params(child_agent.policy.optimizer)
        # distill_lr = 1e-4 # 為蒸餾設定一個合理的學習率
        # child_agent.policy.optimizer = torch.optim.Adam(child_agent.policy.parameters(), lr=distill_lr)
        # 檢查 teacher_means 值域
        blob = torch.load(distill_pt)
        means = blob["teacher_means"]
        print("\nteacher_mean")
        print(means.min().item(), means.max().item())

# 2) 先拍快照
        policy_before = copy.deepcopy(child_agent.policy).cpu()
        print("\n--- 開始執行離線蒸餾預訓練 ---")

        train_network_offline(
            ppo_model=child_agent,
            pt_path=distill_pt,
            optimizer=child_agent.policy.optimizer,
            epochs=5,             # 離線訓練10              # 離線訓練使用稍大的學習率
            vf_coef=0.5,
            device=child_agent.device,
            base_lr=1e-4,
        )
        ok = diff_report_before_after(policy_before, child_agent.policy, atol=1e-12)
        for h in hook_handles:
            h.remove()
        unfreeze_all(child_agent.policy)
        # 2.4 【關鍵】為子代策略設定我們特製的「精細版差異化優化器」
        # 注意：線上微調時，我們可能希望交叉通道的學習率高一些，所以這裡用了 lr_pure/10
        child_agent.policy.optimizer = create_refined_differential_optimizer(
            policy=child_agent.policy,
            lr_pure=child_agent.learning_rate, # 使用 PPO agent 的主學習率
            cross_lr_scale_factor= 0.2625007856712434 # 線上微調時，給予交叉通道 10% 的學習率
        )
        
        # 2.5 執行「全局離線蒸餾」作為熱身 (使用軟硬結合的損失函數)
        
        # 2.6 執行最終的「線上微調」
        print("\n--- 開始執行線上強化學習微調 ---")
        finetune_steps = 100000
        auto_difficulty_callback = AutoDifficultyCallback(
        env,None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=False,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
        child_agent.learn(total_timesteps=finetune_steps, callback=[ auto_difficulty_callback,],progress_bar=True)

        evolved = child_agent # 將訓練完成的 agent 賦值給 evolved 變數
        
        # --- 步驟 3: 儲存訓練好的模型 (這部分保持不變) ---
        print("\n--- 正在儲存融合後的模型 ---")
        out_dir = Path("./models"); out_dir.mkdir(parents=True, exist_ok=True)
        dad_base = os.path.splitext(os.path.basename(dad_path))[0]
        mom_base = os.path.splitext(os.path.basename(mom_path))[0]
        tag = f"child_{dad_base}_x_{mom_base}_fused" # 加上 fused 標籤以區分
        zip_path = out_dir / f"{tag}.zip"
        evolved.save(str(zip_path))
        pkl_path = out_dir / f"{tag}_full.pkl"
        evolved.env = None # 移除 env 以便打包
        with open(pkl_path, "wb") as f:
            # 需要 import cloudpickle
            import cloudpickle
            cloudpickle.dump(evolved, f)

        print("✅ Saved:")
        print(f"   - SB3 zip:   {zip_path}")
        print(f"   - Full .pkl: {pkl_path}")
    elif args.test_all_generations:
        test_all_generations(
        root="./test_models_by_gen",
        hardseed_path="./logs/hard_seeds.json",
        num_samples=1000,
        num_workers=8,
        device_str="cpu"
    )
    elif args.ga:
        # === GA 全局設定 ===
        TOTAL_GENERATIONS = 10                # 要演化幾代
        PARENT_FOLDER = "./best_model/p_models"
        BASE_ENV = "BipedalWalkerCustom-v0"
        FITNESS_PATH = "./logs/ga_eval/fitness_top.json"
        DISTILL_PT = "./logs/ga_eval/mtkd_continuous.pt(2)"

        # === 演化迴圈 ===
        for generation in range(1, TOTAL_GENERATIONS + 1):
            new_id=0
            print(f"\n🧬 ===== Generation {generation} / {TOTAL_GENERATIONS} =====")
            CHILD_SAVE_DIR = Path(f"./child_models/gen{generation}")
            CHILD_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            USED_PAIRS_PATH =f"./logs/ga_eval_gen_ot{generation}/gen{generation}_used_pairs.json"
            os.makedirs(os.path.dirname(USED_PAIRS_PATH), exist_ok=True)
            # Step 1️⃣ 評估當前族群
            model_files = sorted([os.path.join(PARENT_FOLDER, f) for f in os.listdir(PARENT_FOLDER) if f.endswith((".zip",".pkl"))])
            fitness_out, top_names = test_population(
                model_paths=model_files,
                hardseed_path="./logs/hard_seeds.json",
                env_id=BASE_ENV,
                num_samples=100,
                eval_n_stage1=1,
                eval_n_stage2=10,
                reward_threshold=250,
                top_k=10,
                save_dir=f"./logs/ga_eval_gen_ot{generation}",
                save_name=f"mtkd_gen_ot{generation}.pt",
                num_workers=10,
                device_str="cuda",
                diff_min=0.0,
                diff_max=1.0,
                generate_distill=False,
            )
            PAIR_CANDIDATES_PATH=f"./logs/ga_eval_gen{generation}/pair_candidates.json"
            # Step 2️⃣ 讀取 fitness_top.json 模型名單
            with open(f"./logs/ga_eval_gen{generation}/fitness_top.json", "r", encoding="utf-8") as f:
                table = json.load(f)
            names = list(table.keys()) if isinstance(table, dict) else [item["name"] for item in table]

            if len(names) < 2:
                raise ValueError("❌ fitness_top.json 至少需要兩個模型")

            # Step 3️⃣ 若未產生組合，則建立全集
            if not os.path.exists(PAIR_CANDIDATES_PATH):
                all_pairs = [(mom, dad) for mom in names for dad in names if mom != dad]
                with open(PAIR_CANDIDATES_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, indent=2, ensure_ascii=False)
                print(f"✅ 已產生所有父母組合，共 {len(all_pairs)} 組")
            else:
                with open(PAIR_CANDIDATES_PATH, "r", encoding="utf-8") as f:
                    all_pairs = json.load(f)

            # Step 4️⃣ 讀取 used_pairs.json，取得未用組合
            if os.path.exists(USED_PAIRS_PATH):
                with open(USED_PAIRS_PATH, "r", encoding="utf-8") as f:
                    used_pairs = set(tuple(p) for p in json.load(f))
            else:
                used_pairs = set()

            unused_pairs = [p for p in all_pairs if tuple(p) not in used_pairs]
            print(f"📘 尚未交配組合數：{len(unused_pairs)}")

            # Step 5️⃣ 針對每一組父母進行交配（完整遍歷）
            for mom_name, dad_name in unused_pairs:
                print(f"\n➡️ 交配中：mom={mom_name}, dad={dad_name}")
                new_id+=1

                # 標記為已用
                used_pairs.add((mom_name, dad_name))
                with open(USED_PAIRS_PATH, "w", encoding="utf-8") as f:
                    json.dump(sorted(list(map(list, used_pairs))), f, indent=2, ensure_ascii=False)

                # === 路徑解析 ===
                def resolve_path(name: str, root: str) -> str:
                    base = os.path.join(root, name)
                    # ✅ 若有 .pkl 優先
                    if os.path.exists(base + ".pkl"):
                        return base + ".pkl"
                    elif os.path.exists(base + ".zip"):
                        return base + ".zip"
                    elif os.path.exists(base):
                        return base
                    else:
                        # fallback：搜尋整個資料夾
                        for ext in (".pkl", ".zip"):
                            candidate = os.path.join(root, name + ext)
                            if os.path.exists(candidate):
                                return candidate
                    return None

                mom_path = resolve_path(mom_name, PARENT_FOLDER)
                dad_path = resolve_path(dad_name, PARENT_FOLDER)
                env = gym.make(BASE_ENV, difficulty=0.5)
                def safe_load_any(path, env):
                    if path.endswith(".pkl"):
                        with open(path, "rb") as f:
                            policy = cloudpickle.load(f)
                        model = PPO("MlpPolicy", env, verbose=0)
                        model.policy = policy.to(model.device)
                        print(f"✅ Loaded .pkl model: {path}")
                        return model
                    else:
                        try:
                            model = PPO.load(path, env=env, custom_objects={
                                "optimizer": None,
                                "learning_rate": 3e-4,
                                "lr_schedule": lambda _: 3e-4
                            })
                            print(f"✅ Loaded .zip model: {path}")
                            return model
                        except Exception as e:
                            print(f"[❌] Failed to load {path}: {e}")
                            return None
                mom = safe_load_any(mom_path, env=env)
                dad = safe_load_any(dad_path, env=env)

                # === 交配與蒸餾 ===
                child_policy = create_dual_channel_policy(dad.policy, mom.policy)
                if args.crossover == "ties":
                    print("[crossover=ties] 用 TIES 初始化交叉通道")
                    ties_initialize_crosstalk(child_policy, dad.policy, mom.policy, k=args.ties_k)
                elif args.crossover == "ties_progressive":
                    print("[crossover=ties_progressive] 漸進式逐層 TIES：延後到 child_agent 建立後執行")
                    # 融合在 progressive_ties_evolve 內進行（需要 env 逐 stage 微調）
                elif args.crossover == "ot":
                    print("[crossover=ot] 用 OT Fusion 初始化交叉通道")
                    ot_initialize_crosstalk(child_policy, dad.policy, mom.policy)
                elif args.crossover == "ot_recursive":
                    print("[crossover=ot_recursive] 用遞迴 OT 初始化交叉通道")
                    recursive_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, env=env)
                elif args.crossover == "ot_iter":
                    print(f"[crossover=ot_iter] 用迭代 OT 初始化交叉通道（{args.ot_iters} 次）")
                    iterative_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=args.ot_iters)
                elif args.crossover == "ot_mutual":
                    print(f"[crossover=ot_mutual] 兩通道互相 OT 融合（{args.ot_iters} 輪）")
                    mutual_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=max(args.ot_iters, 100), tol=args.ot_tol, obs_batch=_sample_obs_batch(env, n=256))
                elif args.crossover == "ot_progressive":
                    print("[crossover=ot_progressive] 漸進式逐層 OT：延後到 child_agent 建立後執行")
                    # 融合在 progressive_recursive_ot_evolve 內進行（需要 env 逐 stage 微調）
                elif args.crossover == "ot_progressive_iter":
                    print("[crossover=ot_progressive_iter] 漸進式逐層+逐輪迭代 OT：延後到 child_agent 建立後執行")
                    # 融合在 progressive_iterative_ot_evolve 內進行（需要 env 逐 stage、逐輪微調）
                elif args.crossover == "ot_progressive_iter_focused":
                    print("[crossover=ot_progressive_iter_focused] 漸進式逐層+逐輪迭代 OT（聚焦蒸餾版）：延後到 child_agent 建立後執行")
                    # 融合在 progressive_iterative_ot_evolve_focused 內進行
                else:
                    print("[crossover=zero] 用小噪音初始化交叉通道")
                    zero_initialize_crosstalk(child_policy)
                child_agent = PPO(
                    policy=dad.policy.__class__,
                    env=env,
                    learning_rate=4.5e-6,
                    n_steps=dad.n_steps,
                    batch_size=dad.batch_size,
                    n_epochs=dad.n_epochs,
                    gamma=dad.gamma,
                    gae_lambda=dad.gae_lambda,
                    clip_range=dad.clip_range,
                    ent_coef=dad.ent_coef,
                    verbose=0
                )
                child_agent.policy = child_policy.to(child_agent.device)

                # 離線蒸餾：OT 系列的融合值本身就是交叉通道的解，
                # 蒸餾只會用大 lr 把它改寫掉，因此直接跳過
                if args.crossover == "ot_progressive":
                    # 漸進式逐層 OT：每 stage 融合一層交叉通道 + 線上微調穩定
                    progressive_recursive_ot_evolve(
                        child_agent, dad.policy, mom.policy,
                        steps_per_stage=20_000,
                    )
                elif args.crossover == "ties_progressive":
                    # 漸進式逐層 TIES：每 stage 融合一層交叉通道 + 線上微調穩定
                    progressive_ties_evolve(
                        child_agent, dad.policy, mom.policy,
                        k=args.ties_k, steps_per_stage=20_000,
                    )
                elif args.crossover == "ot_progressive_iter":
                    # 漸進式逐層 + 逐輪迭代 OT：每層固定跑 n_rounds 輪 OT→蒸餾後才換下一層
                    progressive_iterative_ot_evolve(
                        child_agent, dad.policy, mom.policy, DISTILL_PT,
                        env=env,
                        n_rounds=args.progressive_n_rounds,
                        final_finetune_steps=args.progressive_steps_per_round,
                        device=str(child_agent.device),
                        alpha_init=args.progressive_ot_alpha_init,
                        alpha_gamma=args.progressive_ot_gamma,
                    )
                elif args.crossover == "ot_progressive_iter_focused":
                    # 跟 ot_progressive_iter 相同，但每輪蒸餾只聚焦剛 OT 對齊過的那一層
                    progressive_iterative_ot_evolve_focused(
                        child_agent, dad.policy, mom.policy, DISTILL_PT,
                        env=env,
                        n_rounds=args.progressive_n_rounds,
                        final_finetune_steps=args.progressive_steps_per_round,
                        device=str(child_agent.device),
                        alpha_init=args.progressive_ot_alpha_init,
                        alpha_gamma=args.progressive_ot_gamma,
                    )
                elif args.crossover in ("ot", "ot_recursive"):
                    print(f"[crossover={args.crossover}] 保留 OT 融合交叉通道，跳過離線蒸餾")
                    unfreeze_all(child_agent.policy)
                else:
                    trainable_params, hook_handles = freeze_pure_channels(child_agent.policy, bias_mode="freeze", allow_train_if_unsplit=False)
                    child_agent.policy.optimizer = torch.optim.Adam(trainable_params, lr=0.008, weight_decay=0.0)
                    train_network_offline(
                        ppo_model=child_agent,
                        pt_path=DISTILL_PT,
                        optimizer=child_agent.policy.optimizer,
                        epochs=5,
                        vf_coef=0.5,
                        device=child_agent.device,
                        base_lr=1e-4,
                    )
                    for h in hook_handles: h.remove()
                    unfreeze_all(child_agent.policy)

                # 線上微調
                auto_difficulty_callback = AutoDifficultyCallback(
                    env, None, eval_freq=10_000, reward_threshold=250,
                    increase=0.05, verbose=1, shared_flags=False,
                    cooldown_steps=0, hardseed_save_path="./logs/hard_seeds.json"
                )
                child_agent.learn(total_timesteps=100000, callback=[auto_difficulty_callback], progress_bar=True)
                # 儲存子代3
                dad_base = os.path.splitext(os.path.basename(dad_path))[0]
                mom_base = os.path.splitext(os.path.basename(mom_path))[0]
                tag = f"gen{generation}_ind{new_id}"
                # zip_path = CHILD_SAVE_DIR / f"{tag}.zip"
                pkl_path = CHILD_SAVE_DIR / f"{tag}.pkl"
                save_lineage(tag, generation, new_id, dad_name, mom_name)
                # child_agent.save(str(zip_path))
                child_agent.env=None
                with open(pkl_path, "wb") as f:
                 cloudpickle.dump(child_agent.policy, f)
                # print(f"✅ 已儲存子代模型：{zip_path}")
                print(f"✅ 已儲存子代模型：{pkl_path}")
            # Step 6️⃣ 評估所有子代 → 選 top-k 進入下一代
            child_models = [os.path.join(CHILD_SAVE_DIR, f)
                        for f in os.listdir(CHILD_SAVE_DIR)
                        if f"gen{generation}_" in f and (f.endswith(".zip") or f.endswith(".pkl"))]
            fitness_out, top_names = test_population(
                model_paths=child_models,
                hardseed_path="./logs/hard_seeds.json",
                env_id=BASE_ENV,
                num_samples=300,
                eval_n_stage1=1,
                eval_n_stage2=10,
                reward_threshold=250,
                top_k=10,
                save_dir=f"./logs/ga_eval_gen{generation}_children",
                save_name=f"children_gen{generation}.pt",
                num_workers=5,
                device_str="cpu",
                diff_min=0.0,
                diff_max=1.0,
                generate_distill=False,
            )

            # Step 7️⃣ 將 top 模型複製為下一代族群
            next_gen_dir = f"./best_model/gen{generation+1}"
            os.makedirs(next_gen_dir, exist_ok=True)

            # 1️⃣ 複製「子代」Top 模型
            for name in top_names:
                src = resolve_path(name, CHILD_SAVE_DIR)
                dst = os.path.join(next_gen_dir, os.path.basename(src))
                shutil.copy2(src, dst)

            # 2️⃣ 複製「父母」Top 模型 (上一代的精英)
            #    你可以根據上一步的 fitness_top.json 取出前幾名
            parent_fitness_path = f"./logs/ga_eval_gen{generation}/fitness_top.json"
            if os.path.exists(parent_fitness_path):
                with open(parent_fitness_path, "r", encoding="utf-8") as f:
                    parent_table = json.load(f)
                parent_top_names = list(parent_table.keys())[:3]   # ← 可調整保留幾個父母，例如 top5

                for pname in parent_top_names:
                    psrc = resolve_path(pname, PARENT_FOLDER)
                    pdst = os.path.join(next_gen_dir, os.path.basename(psrc))
                    if os.path.exists(psrc):
                        shutil.copy2(psrc, pdst)
                        print(f"🌟 保留父母精英：{psrc}")

            print(f"🌱 下一代族群建立完成（包含子代與父母）：{next_gen_dir}")

            # 更新 PARENT_FOLDER 以便下一輪使用
            PARENT_FOLDER = next_gen_dir

        print("\n🎉 GA 演化完成，共產生 3 代！")
    elif args.find_value:
        study = optuna.create_study(
        direction="maximize",
        study_name="BipedalWalker_Fusion_HPO",
        # 您可以將 Optuna 的日誌儲存到資料庫中，以便後續分析
        storage="sqlite:///hpo_results.db", 
        load_if_exists=True
    )

        # 執行優化，例如進行 100 次完整的實驗
        study.optimize(objective, n_trials=100,n_jobs=1)

        # ----------------------------------------------------
        # 3. 輸出最佳結果
        # ----------------------------------------------------
        print("\n\n===== HPO 完成！=====")
        print("最佳 Trial 編號:", study.best_trial.number)
        print("最佳平均獎勵:", study.best_value)
        print("最佳超參數組合:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print("\n--- 正在生成並儲存視覺化分析圖表 ---")
        
        # 創建一個專門存放圖表的資料夾
        output_dir = Path("./hpo_plots")
        output_dir.mkdir(exist_ok=True)
        
        # 檢查是否有已完成的 Trial
        if len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])) > 0:
            
            # --- A) 儲存 Slice Plot ---
            slice_fig = optuna.visualization.plot_slice(study)
            slice_path = output_dir / f"{study.study_name}_slice_plot.html"
            # 使用 .write_html() 儲存為網頁檔案
            slice_fig.write_html(str(slice_path))
            print(f"  - 已儲存 Slice Plot 至: {slice_path}")

            # --- B) 儲存參數重要性圖 ---
            try:
                importance_fig = optuna.visualization.plot_param_importances(study)
                importance_path = output_dir / f"{study.study_name}_importance_plot.html"
                importance_fig.write_html(str(importance_path))
                print(f"  - 已儲存 Importance Plot 至: {importance_path}")
            except Exception as e:
                print(f"\n  - 無法生成參數重要性圖: {e}")
            
            print("\n請用您的網頁瀏覽器開啟以上 .html 檔案來查看互動式圖表。")
        else:
            print("\n沒有已完成的 Trial，無法生成視覺化圖表。")
    
    elif args.test_ties:
        # ── 路徑檢查 ────────────────────────────────────────────────
        if not args.dad or not args.mom:
            print("❌ 請用 --dad 和 --mom 指定兩個模型路徑")
            print("   範例：python BipedalWalker-v3.py --test_ties --dad ./best_model/a.pkl --mom ./best_model/b.pkl")
            exit(1)

        def _load_any(path, env):
            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    policy = cloudpickle.load(f)
                model = PPO("MlpPolicy", env, verbose=0)
                model.policy = policy.to(model.device)
            else:
                # SB3 的 PPO.load 會自動加 .zip，路徑本身已有 .zip 時去掉避免雙重副檔名
                load_path = path[:-4] if path.endswith(".zip") else path
                model = PPO.load(load_path, env=env, device="cpu")
            return model

        ENV_ID = "BipedalWalkerCustom-v0"
        DIFF   = 0.0
        N_EVAL = args.ties_n_eval

        env = gym.make(ENV_ID, difficulty=DIFF, render_mode=None)

        print(f"\n載入 dad：{args.dad}")
        dad = _load_any(args.dad, env)
        print(f"載入 mom：{args.mom}")
        mom = _load_any(args.mom, env)

        # ── 跑 dad/mom baseline ─────────────────────────────────────
        def _eval(model, label, n=N_EVAL):
            rewards = []
            for ep in range(n):
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
            print(f"  [{label}] 平均 = {avg:.2f}\n")
            return avg

        print("\n===== Dad baseline =====")
        dad_avg = _eval(dad, "dad")
        print("===== Mom baseline =====")
        mom_avg = _eval(mom, "mom")

        # ── 雙通道融合 ───────────────────────────────────────────────
        is_progressive = args.crossover in ("ot_progressive", "ot_progressive_iter", "ties_progressive", "ot_first_distill_rest", "distill_only")
        print(f"\n===== 雙通道融合 (crossover={args.crossover}) =====")
        child_policy = create_dual_channel_policy(dad.policy, mom.policy)
        if args.crossover == "ties":
            print(f"[crossover=ties] 用 TIES 初始化交叉通道 (k={args.ties_k})")
            ties_initialize_crosstalk(child_policy, dad.policy, mom.policy, k=args.ties_k)
        elif args.crossover == "ot":
            print(f"[crossover=ot] 用 OT Fusion 初始化交叉通道（ot_frac={args.ot_frac}）")
            ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, ot_frac=args.ot_frac)
        elif args.crossover == "ot_recursive":
            print(f"[crossover=ot_recursive] 用遞迴 OT 初始化交叉通道（ot_frac={args.ot_frac}）")
            recursive_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, env=env, ot_frac=args.ot_frac)
        elif args.crossover == "ot_iter":
            print(f"[crossover=ot_iter] 用迭代 OT 初始化交叉通道（{args.ot_iters} 次，ot_frac={args.ot_frac}）")
            iterative_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=args.ot_iters, ot_frac=args.ot_frac)
        elif args.crossover == "ot_mutual":
            print(f"[crossover=ot_mutual] 兩通道互相 OT 融合（{args.ot_iters} 輪，ot_frac={args.ot_frac}）")
            mutual_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, n_iters=max(args.ot_iters, 100), tol=args.ot_tol, obs_batch=_sample_obs_batch(env, n=256), ot_frac=args.ot_frac)
        elif args.crossover == "ot_reconstruct":
            print("[crossover=ot_reconstruct] 用「重建被截肢半層」初始化交叉通道")
            reconstruct_ot_initialize_crosstalk(child_policy, dad.policy, mom.policy, env=env)
        elif is_progressive:
            print(f"[crossover={args.crossover}] 漸進式融合：延後到 child_model 建立後、邊訓練邊逐層融合")
            # 交叉通道會在下面 progressive_recursive_ot_evolve / progressive_ties_evolve 內處理
        else:
            print("[crossover=zero] 用小噪音初始化交叉通道")
            zero_initialize_crosstalk(child_policy)

        # ── 基本完整性檢查 ──────────────────────────────────────────
        # 漸進式方法此時交叉通道還沒融合（會在訓練中逐層寫入），跳過此刻的數值比對
        if not is_progressive:
            dad_sd   = dad.policy.state_dict()
            mom_sd   = mom.policy.state_dict()
            child_sd = child_policy.state_dict()
            identical_dad = all(torch.equal(child_sd[key], dad_sd[key]) for key in child_sd)
            identical_mom = all(torch.equal(child_sd[key], mom_sd[key]) for key in child_sd)
            print(f"[檢查] child == dad 全部相同？ {identical_dad}  (應為 False)")
            print(f"[檢查] child == mom 全部相同？ {identical_mom}  (應為 False)")
            child_vec = _sd_to_vector(child_sd)
            print(f"[檢查] 參數向量 mean={child_vec.mean():.4f}  std={child_vec.std():.4f}  "
                  f"nan={child_vec.isnan().any().item()}  inf={child_vec.isinf().any().item()}")

        # ── 建立 child model ─────────────────────────────────────────
        child_model = PPO("MlpPolicy", env, verbose=0)
        child_model.policy = child_policy.to(child_model.device)
        child_model.policy.optimizer = torch.optim.Adam(
            child_model.policy.parameters(), lr=1e-4
        )

        # ── 漸進式逐層融合 + 訓練，或蒸餾（兩者互斥）────────────────
        if args.crossover == "ot_progressive":
            if os.path.exists(args.distill_pt):
                print(f"\n===== 蒸餾交叉通道基底 (資料：{args.distill_pt}) =====")
                distill_crosstalk_baseline(child_model, args.distill_pt, epochs=5, lr=0.008, device=child_model.device)
            else:
                print(f"\n[警告] 找不到蒸餾資料 {args.distill_pt}，跳過交叉通道基底蒸餾，直接進入漸進式 OT")
            print(f"\n===== 漸進式逐層 OT（每 stage {args.progressive_steps_per_stage} 步，max_stages={args.progressive_max_stages}，anneal={not args.progressive_no_anneal}）=====")
            progressive_recursive_ot_evolve(
                child_model, dad.policy, mom.policy, env=env,
                steps_per_stage=args.progressive_steps_per_stage,
                max_stages=args.progressive_max_stages,
                anneal=not args.progressive_no_anneal,
                ot_frac=args.ot_frac,
            )
        elif args.crossover == "ot_progressive_iter":
            print(f"\n===== 漸進式逐層+逐輪迭代 OT（每層 {args.progressive_n_rounds} 輪，每輪蒸餾消化，"
                  f"最終微調 {args.progressive_steps_per_round} 步，max_stages={args.progressive_max_stages}）=====")
            progressive_iterative_ot_evolve(
                child_model, dad.policy, mom.policy, args.distill_pt, env=env,
                n_rounds=args.progressive_n_rounds,
                final_finetune_steps=args.progressive_steps_per_round,
                max_stages=args.progressive_max_stages,
                ot_frac=args.ot_frac,
                device=str(child_model.device),
                alpha_init=args.progressive_ot_alpha_init,
                alpha_gamma=args.progressive_ot_gamma,
                pre_finetune_save_path=_pre_finetune_out_path(args),
            )
        elif args.crossover == "ot_progressive_iter_focused":
            print(f"\n===== 漸進式逐層+逐輪迭代 OT（聚焦蒸餾版，每層 {args.progressive_n_rounds} 輪，"
                  f"最終微調 {args.progressive_steps_per_round} 步，max_stages={args.progressive_max_stages}）=====")
            progressive_iterative_ot_evolve_focused(
                child_model, dad.policy, mom.policy, args.distill_pt, env=env,
                n_rounds=args.progressive_n_rounds,
                final_finetune_steps=args.progressive_steps_per_round,
                max_stages=args.progressive_max_stages,
                ot_frac=args.ot_frac,
                device=str(child_model.device),
                alpha_init=args.progressive_ot_alpha_init,
                alpha_gamma=args.progressive_ot_gamma,
                pre_finetune_save_path=_pre_finetune_out_path(args),
            )
        elif args.crossover == "ties_progressive":
            print(f"\n===== 漸進式逐層 TIES（k={args.ties_k}，每 stage 退火 {args.progressive_steps_per_stage} 步）=====")
            progressive_ties_evolve(
                child_model, dad.policy, mom.policy,
                k=args.ties_k, steps_per_stage=args.progressive_steps_per_stage,
            )
        elif args.crossover == "ot_first_distill_rest":
            if os.path.exists(args.distill_pt):
                print(f"\n===== OT 對齊第一層並凍結，其餘層蒸餾適應 (資料：{args.distill_pt}) =====")
                ot_first_layer_then_distill_rest(
                    child_model, dad.policy, mom.policy, args.distill_pt,
                    epochs=5, lr=0.008, device=child_model.device,
                    ot_frac=args.ot_frac,
                )
                # 蒸餾結束後全網已解凍（含 OT 對齊過的第一層），接上真實環境訓練，
                # 讓 PPO 用真實 reward 訊號修正蒸餾殘留的 distributional shift 誤差，
                # 第一層也會跟著一起被微調，不會維持成固定的純 OT 值。
                if args.progressive_steps_per_stage > 0:
                    print(f"\n===== 接上真實環境訓練（{args.progressive_steps_per_stage} 步，含難度自動升級）=====")
                    auto_difficulty_callback = AutoDifficultyCallback(
                        env, None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,
                        shared_flags=None, cooldown_steps=0, hardseed_save_path="./logs/hard_seeds.json",
                    )
                    _rebuild_policy_optimizer(child_model.policy)
                    child_model.learn(
                        total_timesteps=args.progressive_steps_per_stage,
                        callback=[auto_difficulty_callback], progress_bar=True,
                    )
            else:
                print(f"\n[警告] 找不到蒸餾資料 {args.distill_pt}，跳過")
        elif args.crossover == "distill_only":
            if os.path.exists(args.distill_pt):
                print(f"\n===== 對照組：完全不做OT，全部交叉通道只用蒸餾 (資料：{args.distill_pt}) =====")
                distill_crosstalk_baseline(
                    child_model, args.distill_pt, epochs=5, lr=0.008, device=child_model.device,
                )
            else:
                print(f"\n[警告] 找不到蒸餾資料 {args.distill_pt}，跳過")
        elif os.path.exists(args.distill_pt):
            print(f"\n===== 蒸餾 (資料：{args.distill_pt}, epochs=2) =====")
            unfreeze_all(child_model.policy)
            train_network_offline(
                ppo_model=child_model,
                pt_path=args.distill_pt,
                optimizer=child_model.policy.optimizer,
                epochs=0,
                vf_coef=0.5,
                device=child_model.device,
                base_lr=1e-4,
            )
        else:
            print(f"\n[警告] 找不到蒸餾資料 {args.distill_pt}，跳過蒸餾")

        # ── 跑 child ────────────────────────────────────────────────
        # 訓練期間若接了 AutoDifficultyCallback（例如 progressive_iterative_ot_evolve
        # 最後的真實環境微調），reward 超過門檻會直接把這個共用 env 物件的難度往上調——
        # 這裡強制重設回 DIFF，確保 child 跟 dad/mom 是在同一個難度下被評估，比較才公平。
        env.unwrapped.difficulty = DIFF
        print(f"\n===== Child (雙通道 + {args.crossover} 交叉通道) =====")
        child_avg = _eval(child_model, "child")

        # ── 結果摘要 ────────────────────────────────────────────────
        print("=" * 40)
        print(f"  Dad   平均回報：{dad_avg:.2f}")
        print(f"  Mom   平均回報：{mom_avg:.2f}")
        print(f"  Child 平均回報：{child_avg:.2f}  (雙通道 + {args.crossover})")
        better = child_avg > max(dad_avg, mom_avg)
        print(f"  融合後是否超越雙親？ {'✅ Yes' if better else '❌ No'}")
        print("=" * 40)

        # ── 儲存 child ──────────────────────────────────────────────
        out_path = args.ties_out or f"./models/ties_test_child_{args.crossover}.pkl"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        child_model.env = None
        with open(out_path, "wb") as f:
            cloudpickle.dump(child_model.policy, f)
        print(f"\n✅ 已儲存融合模型：{out_path}")
        env.close()

    else:
        parser.print_help()




