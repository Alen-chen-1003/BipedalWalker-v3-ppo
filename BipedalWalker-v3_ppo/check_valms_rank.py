# -*- coding: utf-8 -*-
"""
讀 v2 study 的多種子驗證結果（study.user_attrs["valms::777::100::3"]），依 3 個訓練種子的
平均分數排名。第一名仍是 #18（inter_ot）→ 結束碼 0；第一名換人 → 結束碼 10（要重新調參）；
結果不完整 → 結束碼 2。排名同時寫到 logs/optuna_ot_prog_iter_v2/valms_rank.txt。
"""
import sys, optuna
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)
KEY, EXPECT_BEST, N_SEEDS = "valms::777::100::3", 18, 3
s = optuna.load_study(study_name="ot_prog_iter_v2", storage="sqlite:///./logs/optuna_ot_prog_iter_v2/study.db")
res = s.user_attrs.get(KEY, {})
done = {int(k): v for k, v in res.items() if len(v.get("seeds", {})) >= N_SEEDS}
lines = []
for n, v in sorted(done.items(), key=lambda kv: -kv[1]["score"]):
    sc = [x["score"] for x in v["seeds"].values()]
    lines.append(f"#{n:3d}  平均 {np.mean(sc):7.2f} ± {np.std(sc):5.2f}  ({' / '.join(f'{x:.1f}' for x in sc)})  {s.trials[n].params}")
out = "\n".join(lines)
if len(done) < 5:
    out += f"\n結果不完整：只有 {len(done)} 組參數跑完 {N_SEEDS} 個種子"
    code = 2
else:
    best = max(done, key=lambda n: done[n]["score"])
    code = 0 if best == EXPECT_BEST else 10
    out += f"\n第一名：#{best}  →  " + ("維持 #18，不重新調參" if code == 0 else "第一名換人，重新調參（v3）")
print(out)
open("./logs/optuna_ot_prog_iter_v2/valms_rank.txt", "w", encoding="utf-8").write(out + "\n")
sys.exit(code)
