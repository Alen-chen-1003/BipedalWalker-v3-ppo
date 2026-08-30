import torch
import torch.nn as nn

# 這是我們要測試的函式
@torch.no_grad()
def zero_initialize_crosstalk(policy_or_module):
    # 為了方便測試，我們讓它可以接收單一模塊
    modules_to_iterate = policy_or_module.named_modules() if hasattr(policy_or_module, 'named_modules') else [("", policy_or_module)]
    
    print("正在執行交叉通道歸零...")
    for name, module in modules_to_iterate:
        if isinstance(module, nn.Linear):
            W = module.weight.data
            in_half = W.shape[1] // 2
            out_half = W.shape[0] // 2
            
            if in_half == 0 or out_half == 0: continue

            print(f"處理層: {name}, 形狀: {W.shape}")
            # 右上角 (母 -> 父) 區塊
            cross_mf_block = W[:out_half, in_half:]
            nn.init.normal_(cross_mf_block, mean=0.0, std=0.001)

            # 選中左下角 (父 -> 母) 區塊
            cross_fm_block = W[out_half:, :in_half]
            nn.init.normal_(cross_fm_block, mean=0.0, std=0.001)
    print("歸零完成。")

# --- 數值模擬開始 ---
# 1. 準備一個模擬層，權重全為 1
mock_layer = nn.Linear(in_features=4, out_features=6)
with torch.no_grad():
    mock_layer.weight.data.fill_(1.0)

print("--- 操作前的權重矩陣 W ---")
print(mock_layer.weight.data)
print("-" * 30)

# 2. 對這個模擬層執行我們的函式
zero_initialize_crosstalk(mock_layer)

print("-" * 30)
print("--- 操作後的權重矩陣 W ---")
print(mock_layer.weight.data)