import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual simulation results)
# Each point represents a different strategy or parameter setting
# Lower RMSE and lower Communication Load is better

# RMSE values (tracking performance)
rmse_periodic = 3.2
rmse_etc = 3.4
rmse_etc_aggressive = 3.6
rmse_etc_conservative = 3.1

# Communication load values (e.g., total number of bytes transmitted)
comm_periodic = 100000  # in bytes
comm_etc = 60000
comm_etc_aggressive = 40000
comm_etc_conservative = 80000

# Organize into arrays
rmses = np.array([rmse_periodic, rmse_etc, rmse_etc_aggressive, rmse_etc_conservative])
comm_loads = np.array([comm_periodic, comm_etc, comm_etc_aggressive, comm_etc_conservative])
labels = ['Periodic', 'ETC', 'ETC (Aggressive)', 'ETC (Conservative)']
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

# Plotting
plt.figure(figsize=(8, 6))
for i in range(len(rmses)):
    plt.scatter(comm_loads[i], rmses[i], color=colors[i], label=labels[i], s=100)
    plt.text(comm_loads[i] * 1.01, rmses[i], labels[i], fontsize=10, va='center')

plt.xlabel("Communication Load [bytes]", fontsize=12)
plt.ylabel("Tracking RMSE [m]", fontsize=12)
plt.title("Trade-off between Communication Load and Tracking Performance", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

