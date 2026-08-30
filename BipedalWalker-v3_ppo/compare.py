import matplotlib.pyplot as plt
import numpy as np

# Data from the table
frequency = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1e6, 2e6, 5e6]

# Gain values in dB for each VDD
# Note: The table shows 0 dB for Vout=0 at low frequencies, which is inconsistent.
# A Vout of 0 should technically be -infinity dB.
# The plot below uses the exact numbers from your table's dB columns.
gain_db_0_9v = [0, 0, -3.77, -3.77, 6.02, 13.62, 17.6, 21.0, 21.6, 21.9, 21.9, 21.9, 21.9, 20.9, 18.5, 14.6, 8.3]
# Assuming the middle column is for VDD = 1.0V
gain_db_1_0v = [0, 0, -3.88, -3.77, 6.02, 13.62, 17.62, 20.98, 21.66, 21.87, 21.87, 21.87, 21.87, 20.91, 18.49, 14.65, 8.3]
# The '<-' in the table for 1.1V at 20Hz and 50Hz are interpreted as having no valid output, so we start plotting from 100Hz for this line.
freq_1_1v = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1e6, 2e6, 5e6]
gain_db_1_1v = [-3.1, 0, 6.02, 11.32, 16.65, 20.17, 22.54, 22.41, 22.54, 21.29, 21.29, 20.34, 19.2, 13.44, 9.19]


# Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))

plt.plot(frequency, gain_db_0_9v, 'o-', label='VDD = 0.9V', color='blue')
plt.plot(frequency, gain_db_1_0v, 's-', label='VDD = 1.0V', color='green')
plt.plot(freq_1_1v, gain_db_1_1v, '^-', label='VDD = 1.1V', color='red')

# Formatting the plot
plt.xscale('log')
plt.title('Bode Plot for Different VDD', fontsize=16)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Gain (Vout/Vsig, dB)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")

# Show the plot
plt.show()
