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

avgNodes, avgTime = [], []

# Load target data - TO DOWNSAMPLE
for i in range(targetNum):
    target_x_traj[:,i] = np.loadtxt(log_directory+'/target_x_traj'+str(i+1)+'.txt')
    target_y_traj[:,i] = np.loadtxt(log_directory+'/target_y_traj'+str(i+1)+'.txt')

for i in range(int(auvNum)):
    # AUVs Simulation Data - TO DOWNSAMPLE
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv_x_traj'+str(i+1)+'.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv_y_traj'+str(i+1)+'.txt')

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    


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


import numpy as np
import matplotlib.pyplot as plt


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

###########################################################
import numpy as np
import matplotlib.pyplot as plt

def compute_fiedler_value_snr(positions, SL, NL, DI, alphaFunc, f, config):
    """
    Computes the Fiedler value of the AUV network using an SNR-based adjacency matrix.
    """
    num_nodes = positions.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d_ij = np.linalg.norm(positions[i] - positions[j])
            if d_ij == 0:
                continue

            TL = 20 * np.log10(d_ij) + (d_ij * alphaFunc(f) * 1e-3)
            snr_ij = SL - TL - NL + DI

            if config.SNR_lb < snr_ij < config.SNR_ub:
                weight = -snr_ij / config.SNR_ub
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # symmetric

    degree_matrix = np.diag(np.abs(adj_matrix).sum(axis=1))
    laplacian = degree_matrix - adj_matrix

    eigenvalues = np.linalg.eigvalsh(laplacian)
    return np.sort(eigenvalues)[1] if len(eigenvalues) > 1 else 0


def plot_fiedler_trend_snr(auv_x_traj, auv_y_traj, SL, NL, DI, alphaFunc, f, config):
    """
    Plots the Fiedler value over time using SNR-based adjacency matrix.
    """
    samples = auv_x_traj.shape[0]
    t = np.linspace(0,simTime,samples)
    print(samples)
    fiedler_vals, zeros = [], []

    for t in range(samples):
        positions = np.vstack((auv_x_traj[t], auv_y_traj[t])).T  # shape: (auvNum, 2)
        fiedler_val = compute_fiedler_value_snr(positions, SL, NL, DI, alphaFunc, f, config)
        fiedler_vals.append(fiedler_val)
        zeros.append(0.0)

    
    plt.figure(figsize=(10, 4))
    plt.plot(fiedler_vals, linewidth=2, color='tab:blue')
    plt.plot(zeros, linewidth=2, color='red', linestyle='--')
    plt.title("Fiedler Value Over Time (SNR-Based Connectivity)")
    plt.xlabel("Time Step")
    plt.ylabel("Fiedler Value (λ₂)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:

# Example config object
class config:
    SNR_lb = 100  # in dB
    SNR_ub = 141

#  absorption function
def alpha_f(f):
    ''' Compute the term alpha(f) according to Stojanovic'09'''
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

# Acoustic parameters
SL = 186  # dB, source level
NL = 30   # dB, noise level
DI = 0    # dB, directivity index
f = 10  # kHz, frequency

# Then call the plot function
plot_fiedler_trend_snr(auv_x_traj, auv_y_traj, SL, NL, DI, alpha_f, f, config)





