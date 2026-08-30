import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

freq=[20,50,100,200,500,1e3,2e3,5e3,10e3,20e3,50e3,100e3,200e3,500e3,1e6,2e6,5e6]
dB0=[5.10,6.02,10.10,12.86,19.46,21.86,22.67,22.92,22.54,22.54,22.27,22.14,20.98,13.62,6.84,6.84,6.02]
dB25=[5.10,6.02,10.62,14.64,20,22.27,23.16,23.52,23.04,23.04,23.04,23.04,21.58,13.97,6.84,6.84,6.02]
dB5=[5.10,6.84,8.94,14.32,20,22.27,23.16,23.52,23.16,23.04,23.04,22.92,21.58,12.86,6.02,6.02,6.02]
dB10=[5.10,6.02,10.10,12.86,19.46,21.86,22.67,22.92,22.54,22.54,22.27,22.14,20.98,13.62,6.84,6.02,6.02]

plt.figure(figsize=(9,6))
plt.semilogx(freq,dB0,'o-',label="R21=0Ω")
plt.semilogx(freq,dB25,'o-',label="R21=2.5kΩ")
plt.semilogx(freq,dB5,'o-',label="R21=5kΩ")
plt.semilogx(freq,dB10,'o-',label="R21=10kΩ")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.title("Bode Plot: Vout/Vsig vs Frequency")
plt.grid(True, which="both")
plt.legend()
plt.show()


