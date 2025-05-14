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

###########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse

def plot_confidence_ellipses(covariance_data):
    """
    Plots confidence ellipses based on the covariance matrices, spreading them out in time.
    
    Args:
    - covariance_data: List of covariance matrices
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Total number of covariance samples
    total_samples = len(covariance_data)

    for i, cov_matrix in enumerate(covariance_data):
        # Eigenvalue decomposition of the covariance matrix
        vals, vecs = np.linalg.eigh(cov_matrix[:2, :2])  # Only the position covariance (2x2)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Calculate the width and height of the ellipse (2 standard deviations)
        width, height = 2 * np.sqrt(vals)

        # Compute the angle of the ellipse
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Distribute the ellipses along the x-axis to simulate time progression
        center_x = 0#i * 0.5#hange this factor to adjust spacing
        center_y = 0  # Keep the y-coordinate constant, or adjust as needed

        # Plot the ellipse with a border
        ellipse = Ellipse(xy=(center_x, center_y), width=width, height=height, angle=angle, 
                          edgecolor='black', facecolor='blue', alpha=0.2, linewidth=2)
        ax.add_patch(ellipse)

    ax.set_title("Confidence Ellipses for Tracking Errors Over Time")
    ax.set_xlabel("X Error (m)")
    ax.set_ylabel("Y Error (m)")
    ax.axis('equal')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_velocity_confidence_ellipses(covariance_data):
    """
    Plots confidence ellipses based on the covariance matrices for estimated velocities.
    
    Args:
    - covariance_data: List of covariance matrices
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Total number of covariance samples
    total_samples = len(covariance_data)

    for i, cov_matrix in enumerate(covariance_data):
        # Extract the 2x2 velocity covariance matrix (vx, vy)
        vel_cov_matrix = cov_matrix[2:4, 2:4]

        # Eigenvalue decomposition of the velocity covariance matrix
        vals, vecs = np.linalg.eigh(vel_cov_matrix)  # Velocity covariance (2x2)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Calculate the width and height of the ellipse (2 standard deviations)
        width, height = 2 * np.sqrt(vals)

        # Compute the angle of the ellipse
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Distribute the ellipses along the x-axis to simulate time progression
        center_x = 0#i * 2  # Change this factor to adjust spacing
        center_y = 0  # Keep the y-coordinate constant, or adjust as needed

        # Plot the ellipse with a border
        ellipse = Ellipse(xy=(center_x, center_y), width=width, height=height, angle=angle, 
                          edgecolor='black', facecolor='red', alpha=0.2, linewidth=2)
        ax.add_patch(ellipse)

    ax.set_title("Confidence Ellipses for Velocity Estimation Errors Over Time")
    ax.set_xlabel("VX Error (m/s)")
    ax.set_ylabel("VY Error (m/s)")
    ax.axis('equal')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage: Plot the confidence ellipses based on loaded covariance data
for i in range(auvNum):
    print(covariance[i])
    plot_confidence_ellipses(covariance[i])
    plot_velocity_confidence_ellipses(covariance[i])

