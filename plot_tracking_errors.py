#!+usr+bin+env python
import matplotlib.pyplot as plt
import os, importlib.util, pathlib
import numpy as np
from math import atan2
import os, pathlib
from scipy.interpolate import make_interp_spline

# Environment initialization
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/uwmsn-sim'
log_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/logs'
class_directory = pkg_directory+'/src'+'/Classes'

# Import config file 
module_dir = os.path.dirname(pathlib.Path(__file__).parent.resolve())
header_file = pkg_directory+'/include'+'/uwmsn-sim'
spec = importlib.util.spec_from_file_location("module.header", header_file+'/main-kinematic_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

# Load simulation info
sim_info = np.loadtxt(log_directory+'/sim_info.txt')
sim_data = np.loadtxt(log_directory+'/sim_data.txt')
ctrl_set = np.loadtxt(log_directory+'/ctrl_set.txt')

print('Simulation info: [Slot Time (TDMA), Minimum Distance to the target (m), alpha_w, gamma_w, Dthresh]', sim_info)
auvNum = int(sim_data[0])
targetNum = int(sim_data[1])
simTime = int(sim_data[2])
samples = int(sim_data[3])
# Initialize data structures for AUVs data
auv_x_traj = np.zeros(((samples),(auvNum)))
auv_y_traj = np.zeros(((samples),(auvNum)))
target_x_traj = np.zeros(((samples),(auvNum)))
target_y_traj = np.zeros(((samples),(auvNum)))



guidanceETC = [[] for _ in range((auvNum))]
surge_vel = [[] for _ in range((auvNum))]
heading = [[] for _ in range(auvNum)]
x_hat, tracking_errors, P, avgNodes, avgTime = [], [], [], [], []
PDR = np.zeros(((auvNum),1))

# Load target data - TO DOWNSAMPLE
for i in range(1):
    target_x_traj[:,i] = np.loadtxt(log_directory+'/target_x_traj'+str(i+1)+'.txt')
    target_y_traj[:,i] = np.loadtxt(log_directory+'/target_y_traj'+str(i+1)+'.txt')

covariance = [[] for _ in range(auvNum)]

tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_1.txt')
n_estimations = len(tmp)


for i in range(int(auvNum)):
    # AUVs Simulation Data - TO DOWNSAMPLE
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv_x_traj'+str(i+1)+'.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv_y_traj'+str(i+1)+'.txt')

    guidanceETC[i] = np.loadtxt(log_directory+'/'+str(i+1)+'etcGuidance.txt')
    #surge_vel[i] = np.loadtxt(log_directory+'/'+str(i+1)+'surge_vel')
    #heading[i] = np.loadtxt(log_directory+'/'+str(i+1)+'heading')*180/np.pi

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

    #

    # Estimation Data
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-trackErr.txt')
    tracking_errors.append(err)

    x_hat_j = np.zeros((n_estimations,4))

    #for j in range(auvNum):
    x_hat_j = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_1.txt')
    if len(x_hat_j) < n_estimations:
        n_estimations = len(x_hat_j)     
    x_hat.append(x_hat_j[:n_estimations,:])

    covariance[i] = np.load(log_directory+'/'+str(i+1)+'-cov'+str(targetNum)+'.npy', allow_pickle=True)
    #P.append(cov)

# Downsampling script
original_samples = samples
sampling = 20
samples = int(np.ceil(original_samples / sampling))

# Downsample target positions
target_x = target_x_traj[::sampling, :targetNum]
target_y = target_y_traj[::sampling, :targetNum]

# Downsample AUV positions and other properties
auv_x = auv_x_traj[::sampling, :auvNum]
auv_y = auv_y_traj[::sampling, :auvNum]
surge_vel = [vel[::sampling] for vel in surge_vel[:auvNum]]
heading = [head[::sampling] for head in heading[:auvNum]]


# Initialize data structures with predefined sizes
a_x = [[] for _ in range(auvNum)]
a_y = [[] for _ in range(auvNum)]
l_x = [[] for _ in range(auvNum)]
l_y = [[] for _ in range(auvNum)]
dist = [[] for _ in range(auvNum)]
dist1, dist2,dist3 = [], [], []
t_x = [[] for _ in range(targetNum)]
t_y = [[] for _ in range(targetNum)]
phi_lists = np.zeros((auvNum, samples, 2))  # Structured as (AUVs, samples, 2 angles)

phi = np.zeros((4, 2))
list_phi = np.zeros(samples)



##########################################################
# PLOT SETUP
fs = 50
lw = 8
sw = 1
tw = 40
lwsw = 200
ticks_size = 50
ms = 15
math_vars = ['\\kappa(\\Phi)','\\xi','C^{(d)}+C^{(g)}']

agents_vars = ['s_1','s_2','s_3','s_4']
time_vars = ['(t_{0})','(t_{f})','(t_{0}=t_{f})']


# --- CONFIG ---
lw = 3  # line width
ms = 6  # marker size
fs = 30  # font size

# Example color function (you can replace this with your own)
def colors(i):
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    return color_list[i % len(color_list)]

# --- INPUT ---

failed_auv = 3  # if AUV 2 failed, index is 1
FAILURE = True  # Set True/False depending on simulation

# Active AUVs
if FAILURE:
    active_auvs = [i for i in range(auvNum) if i != failed_auv - 1]
else:
    active_auvs = list(range(auvNum))

active_auvNum = len(active_auvs)

# Plotting
fig, axes = plt.subplots(active_auvNum, 1, figsize=(10, 3.5 * active_auvNum), sharex=True)
if active_auvNum == 1:
    axes = [axes]  # Make iterable

math_vars = ['AUV1', 'AUV2', 'AUV3', '\\epsilon']
t = np.linspace(0, simTime, n_estimations)
epsi = np.zeros_like(t) + 15.0  # 15 m reference error

for idx, i in enumerate(active_auvs):
    # Plot tracking error
    err = tracking_errors[i][:n_estimations]
    axes[idx].plot(t, err, label=rf'$ {math_vars[i]} $',
                   linewidth=lw, marker='o', markersize=ms, color=colors(i))

    # Plot reference threshold
    axes[idx].plot(t, epsi, 'r:', linewidth=lw * 2 / 3, label=rf'$ {math_vars[-1]} $')

    # Plot confidence band (±2σ) using position covariance
    std_list = []
    for cov in covariance[i][:n_estimations]:
        P_pos = cov[:2, :2]
        std = np.sqrt(np.trace(P_pos))  # A simple scalar std proxy
        std_list.append(10*np.sqrt(std))

    std_array = np.array(std_list)
    upper = err + 2 * std_array
    lower = err - 2 * std_array

    axes[idx].fill_between(t, lower, upper, color=colors(i), alpha=0.2, label='±2σ interval')

    # Labels and legend
    if idx == 1 or (active_auvNum == 1 and idx == 0):
        axes[idx].set_ylabel('            RMSE (m)', fontsize=42)
    
    axes[idx].legend(fontsize=25)
    axes[idx].grid()
    axes[idx].tick_params(axis='y', labelsize=fs * 2 / 3)
    axes[idx].tick_params(axis='x', labelsize=fs * 2 / 3)

axes[-1].set_xlabel('t (s)', fontsize=42)
plt.xticks(fontsize=fs * 2 / 3)
plt.tight_layout()


import matplotlib.pyplot as plt

# --- CONFIG ---
fs = 30  # font size
lw = 3   # line width
ms = 6   # marker size

# Active AUVs
if FAILURE:
    active_auvs = [i for i in range(auvNum) if i != failed_auv - 1]
else:
    active_auvs = list(range(auvNum))

active_auvNum = len(active_auvs)

# Collect errors into a single list for boxplot
boxplot_data = [tracking_errors[i][:n_estimations] for i in active_auvs]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Create boxplot
bp = ax.boxplot(boxplot_data, patch_artist=True, widths=0.6)

# Coloring the boxes
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors(active_auvs[i]))
    patch.set_edgecolor('black')
    patch.set_linewidth(2)

# Coloring whiskers and medians
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=2)
for median in bp['medians']:
    median.set(color='darkred', linewidth=3)

# Set axis labels and ticks
ax.set_xticks(np.arange(1, active_auvNum + 1))
ax.set_xticklabels([f'AUV{i+1}' for i in active_auvs], fontsize=fs)
ax.set_ylabel('Tracking RMSE (m)', fontsize=fs)
ax.tick_params(axis='y', labelsize=fs * 2 / 3)
ax.grid(True, linestyle='--')

plt.tight_layout()
plt.show()


plt.show()


