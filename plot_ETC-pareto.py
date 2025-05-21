import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual simulation results)
# Each point represents a different strategy or parameter setting
# Lower RMSE and lower Communication Load is better

# PLOT SETUP
fs = 35
lw = 8
sw = 1
tw = 40
lwsw = 200
ticks_size = 22
ms = 15

# RMSE values (tracking performance)

rmse_etc1 = 8.830068480020698
rmse_etc2 = 7.67714995419153
rmse_etc3 = 6.582400044222864
rmse_etc4 = 6.786532145569878

# Communication load values (e.g., total number of bytes transmitted)

comm_etc1 = 1941
comm_etc2 = 3285
comm_etc3 = 4629
comm_etc4 = 7168

# Organize into arrays
rmses = np.array([rmse_etc1, rmse_etc2, rmse_etc3, rmse_etc4])
comm_loads = np.array([comm_etc1, comm_etc2, comm_etc3, comm_etc4])
labels = ['\\alpha=0.001','\\alpha=0.05','\\alpha=0.1','\\alpha=0.2']
#colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

# Plotting
plt.figure(figsize=(8, 6))
for i in range(len(rmses)):
    plt.scatter(comm_loads[i], rmses[i], label=r'$ %s $'%labels[i], s=500)

plt.xlabel("Communication Load (bit)", fontsize=fs)
plt.ylabel("Tracking RMSE (m)", fontsize=fs)
#plt.title("Communication Load vs Tracking Performance", fontsize=fs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.legend(fontsize=fs*2/3)
plt.yticks(fontsize=ticks_size, rotation= 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=ticks_size, rotation= 0)#to set dimension and orientation of tick labels
plt.show()

