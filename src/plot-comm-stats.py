import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def alpha_f(f):
    '''Compute the term alpha(f) according to Stojanovic'09, input in kHz'''
    return 0.11*(f**2/(1+f**2)) + 44*(f**2/(4100+f**2)) + (2.75e-4)*(f**2) + 0.003

# Constants
TM = 2     # measurement sampling period (s)
SL = 186   # Source Level in dB
NL = 30    # Noise Level in dB
DI = 0     # Directivity index
c = 1500   # speed of sound in water (m/s)
PDR = 90   # Packet Delivery Ratio (%)

# Ranges to explore
n_m_values = np.arange(1, 10)          # Number of transmitted measurements
f_values = np.linspace(2, 10, 100)     # Frequency range from 2 to 10 kHz

# Create grid
n_m_grid, f_grid = np.meshgrid(n_m_values, f_values)

# Compute bandwidth and packet size
pktSize = 1024 + 128 * n_m_grid             # Packet size in bits
B = (f_grid * 1000) / 20                    # Bandwidth in bps (20 bps/Hz efficiency)

# Compute slot time
Ts = TM + np.floor(pktSize / B)            # TDMA slot time in seconds

# Plotting
fig = plt.figure(figsize=(14, 6))

# 3D Surface Plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(n_m_grid, f_grid, Ts, cmap='viridis')
ax1.set_xlabel('Number of measurements $n_m$')
ax1.set_ylabel('Modem frequency $f$ (kHz)')
ax1.set_zlabel('Time slot $T_s$ (s) considering PAM sensors prcoessing')
ax1.set_title('TDMA Slot Time vs Measurements and Frequency')

# Contour Plot
ax2 = fig.add_subplot(1, 2, 2)
cp = ax2.contourf(n_m_grid, f_grid, Ts, cmap='viridis', levels=30)
cbar = plt.colorbar(cp)
ax2.set_xlabel('Number of measurements $n_m$')
ax2.set_ylabel('Modem frequency $f$ (kHz)')
ax2.set_title('Contour of TDMA Slot Time $T_s$')
cbar.set_label('$T_s$ (s)')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Example TDMA slot times (in seconds)

# 1. Fix the number of measurements (the packet size)
# 2. Decrease the bps (or freq of the modem) in each test --> TDMA increase
# I theory you should test from 10kHz to 2 kHz (long range modems), but I expect to low bandwidt a t some point for a packet

Ts_values = np.array([4, 6, 8, 10, 12, 14])

# Example RMSE results from multiple runs per Ts (synthetic data here)
# You should replace these lists with your actual RMSE results from simulations
rmse_per_Ts = {

    4:  [22.4, 19.5, 18.2],
    6:  [31.5, 29.8, 27.4],
    8:  [28.15, 30.30, 27.25],
    10: [42.5, 37.5, 36.5],
    12: [62.5, 57.5, 70.5],
    14: [124, 110, 112],
    

}

# Compute mean and standard deviation of RMSE for each Ts
mean_rmse = []
std_rmse = []

for Ts in Ts_values:
    values = rmse_per_Ts[Ts]
    mean_rmse.append(np.mean(values))
    std_rmse.append(np.std(values))

# Plotting
# Plot setup
fs = 40
lw = 5    
sw = 400
ts = 35
plt.figure(figsize=(8, 6))
plt.errorbar(Ts_values, mean_rmse, yerr=std_rmse, fmt='o-', capsize=5, lw=lw, color='steelblue', ecolor='gray')
plt.xlabel('TDMA Slot Time $T_s$ (s)', fontsize=fs)
plt.ylabel('Mean RMSE (m)', fontsize=fs)
#plt.title('Tracking Performance vs TDMA Slot Time', fontsize=fs)
plt.grid(True)
plt.xticks(Ts_values)
plt.tick_params(axis='both', which='major', labelsize=ts)
#plt.tight_layout()
plt.show()
