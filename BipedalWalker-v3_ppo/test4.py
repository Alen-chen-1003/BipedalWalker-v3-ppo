import matplotlib.pyplot as plt
import numpy as np

# 頻率資料 (Hz)
freq = np.array([20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4,
                 1e5, 2e5, 5e5, 1e6, 2e6, 5e6])

# 增益 (dB)
gain_09 = np.array([0, 1.59, 1.59, 9.54, 13.98, 19.28, 19.83, 20, 20, 20,
                    19.83, 19.09, 17.62, 13.63, 8.30, 5.11, 0])
gain_10 = np.array([0, 0, 0, 9.54, 16.12, 18.06, 18.49, 18.49, 18.69, 18.69,
                    18.69, 18.69, 17.62, 13.63, 8.30, 5.11, 0])
gain_11 = np.array([0, 4.08, 4.08, 10.10, 16.12, 17.15, 18.06, 18.06, 18.06,
                    18.06, 18.06, 18.06, 14.96, 10.10, 4.08, 4.08, 0])

# 畫圖函式
def plot_bode(freq, gain, title, AM, f3dB, ft):
    plt.figure(figsize=(7, 5))
    plt.semilogx(freq, gain, marker='o', linewidth=2)
    plt.title(f"{title}\n$A_M$={AM} dB, $f_{{3dB}}$={f3dB/1e3:.0f} kHz, $f_t$={ft/1e6:.1f} MHz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.axhline(AM - 3, color='r', linestyle='--', label='-3 dB line')
    plt.axvline(f3dB, color='g', linestyle='--', label='$f_{3dB}$')
    plt.axvline(ft, color='purple', linestyle='--', label='$f_t$')
    plt.legend()
    plt.show()

# 三張圖分別畫出
plot_bode(freq, gain_09, "Bode Plot @ 0.9VDD", AM=20, f3dB=1.8e5, ft=3e6)
plot_bode(freq, gain_10, "Bode Plot @ VDD",   AM=18.7, f3dB=3e5, ft=3e6)
plot_bode(freq, gain_11, "Bode Plot @ 1.1VDD", AM=18, f3dB=1.8e5, ft=2e6)
