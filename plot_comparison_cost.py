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
PDR = np.zeros(((auvNum),1))

# Load target data - TO DOWNSAMPLE
for i in range(1):
    target_x_traj[:,i] = np.loadtxt(log_directory+'/target_x_traj'+str(i+1)+'.txt')
    target_y_traj[:,i] = np.loadtxt(log_directory+'/target_y_traj'+str(i+1)+'.txt')

for i in range(int(auvNum)):
    # AUVs Simulation Data - TO DOWNSAMPLE
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv_x_traj'+str(i+1)+'.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv_y_traj'+str(i+1)+'.txt')

    guidanceETC[i] = np.loadtxt(log_directory+'/'+str(i+1)+'etcGuidance.txt')

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

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

# Example config object
class config:
    SNR_lb = 100  # in dB
    SNR_ub = 141

#  absorption function
def alphaFunc(f):
    ''' Compute the term alpha(f) according to Stojanovic'09'''
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

# Acoustic parameters
SL = 186  # dB, source level
NL = 30   # dB, noise level
DI = 0    # dB, directivity index
f = 10  # kHz, frequency


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

# === CONFIGS ===
fs = 50
lw = 8
ticks_size = 50

# === DATA DOWNSAMPLING ===
target_x = target_x_traj[::sampling, :targetNum]
target_y = target_y_traj[::sampling, :targetNum]
auv_x = auv_x_traj[::sampling, :auvNum]
auv_y = auv_y_traj[::sampling, :auvNum]
samples = auv_x.shape[0]

# === COST 1: GEOMETRY-BASED COST J^{(g)} ===
phi_lists = np.zeros((auvNum, samples, 2))
list_phi = np.zeros(samples)

for k in range(targetNum):
    tmp_t_x = target_x[:, k]
    tmp_t_y = target_y[:, k]
    for j in range(auvNum):
        tmp_x = auv_x[:, j]
        tmp_y = auv_y[:, j]
        for i in range(samples):
            angle = atan2(tmp_y[i] - tmp_t_y[i], tmp_x[i] - tmp_t_x[i])
            phi_lists[j, i] = [np.sin(angle), -np.cos(angle)]

'''once = False
for i in range(samples):
    if header.config.AUV_failure and i >= samples / 2:
        if not once:
            phi_lists = np.delete(phi_lists, 0, axis=0)
            once = True
        phi = np.zeros((auvNum-1, 2))
        phi[:] = phi_lists[:, i]
    else:
        phi = np.zeros((auvNum, 2))
        phi[:] = phi_lists[:auvNum, i]

    cost_g = header.utils.computeCost(phi)
    list_phi[i] = cost_g  # J^{(g)}'''

once = False
list_phi = np.zeros(samples)
fiedler_vals = []
distance_cost = []

auvID_failed = 3

for i in range(samples):
    # === Handle AUV failure at runtime ===
    if header.config.AUV_failure and i >= samples / 2:
        if not once:
            phi_lists = np.delete(phi_lists, auvID_failed-1, axis=0)
            auv_x = np.delete(auv_x, auvID_failed-1, axis=1)
            auv_y = np.delete(auv_y, auvID_failed-1, axis=1)
            once = True
        current_auvNum = auvNum - 1
    else:
        current_auvNum = auvNum

    # === Geometry-based cost J^{(g)} ===
    phi = np.zeros((current_auvNum, 2))
    phi[:] = phi_lists[:current_auvNum, i]
    list_phi[i] = header.utils.computeCost(phi)

    # === Connectivity-based cost J^{(c)} (Fiedler) ===
    positions = np.vstack((auv_x[i], auv_y[i]))[:current_auvNum].T
    fiedler_val = compute_fiedler_value_snr(positions, SL, NL, DI, alphaFunc, f, header.config)
    fiedler_vals.append(fiedler_val)

    # === Distance-based cost J^{(d)} ===
    d_cost = 0
    for j in range(current_auvNum):
        for k in range(targetNum):
            d = np.sqrt((auv_x[i, j] - target_x[i, k])**2 + (auv_y[i, j] - target_y[i, k])**2)
            d_cost += d
    distance_cost.append(d_cost)

# === NORMALIZATION (OPTIONAL) ===
list_phi = np.array(list_phi)
fiedler_vals = np.array(fiedler_vals)
distance_cost = np.array(distance_cost)

# Normalize to [0,1] for visual comparison
Jg_norm = list_phi / max(list_phi)
Jc_norm = fiedler_vals / max(fiedler_vals) if np.max(fiedler_vals) > 0 else fiedler_vals
Jd_norm = distance_cost / max(distance_cost)

# === PLOT ===
t_axis = np.linspace(0, simTime, samples)
plt.figure(figsize=(16, 8))
plt.plot(t_axis, Jg_norm, label=r'$J^{(g)}$ (Geometry Cost)', linewidth=lw)
plt.plot(t_axis, Jd_norm, label=r'$J^{(d)}$ (Distance Cost)', linewidth=lw)
plt.plot(t_axis, Jc_norm, label=r'$J^{(c)}$ (Connectivity - Fiedler)', linewidth=lw)

plt.xlabel('Time (s)', fontsize=fs)
plt.ylabel('Normalized Cost Values', fontsize=fs)
plt.legend(fontsize=fs * 0.7)
plt.grid(True)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.tight_layout()
plt.show()