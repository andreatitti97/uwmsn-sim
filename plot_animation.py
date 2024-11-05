#!+usr+bin+env python
import matplotlib.pyplot as plt
import numpy as np
from math import atan2, sin, cos
import os, importlib.util, pathlib
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

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

surge_vel = [[] for _ in range((auvNum))]
heading = [[] for _ in range(auvNum)]
x_hat, tracking_errors, P, avgNodes, avgTime = [], [], [], [], []
PDR = np.zeros(((auvNum),1))

# Load target data - TO DOWNSAMPLE
for i in range(1):
    target_x_traj[:,i] = np.loadtxt(log_directory+'/target_x_traj'+str(i+1)+'.txt')
    target_y_traj[:,i] = np.loadtxt(log_directory+'/target_y_traj'+str(i+1)+'.txt')

# AUVs Simulation Data - TO DOWNSAMPLE
for i in range(auvNum):
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv_x_traj'+str(i+1)+'.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv_y_traj'+str(i+1)+'.txt')

    surge_vel[i] = np.loadtxt(log_directory+'/'+str(i+1)+'surge_vel.txt')
    heading[i] = np.loadtxt(log_directory+'/'+str(i+1)+'heading.txt')*180/np.pi

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

    # Estimation Data
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-trackErr.txt')
    tracking_errors.append(err)
    '''TODO Add the possibility of monitoring all the estimations
    tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_2.txt')
    x_hat_ = np.zeros((len(tmp),4))
    cov = np.zeros((len(tmp),4))
    for j in range(4):
        x_hat_[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_'+str(j+1)+'.txt')
        cov[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-cov'+str(j+1)+'.txt')
    x_hat.append(x_hat_)
    P.append(cov)'''
    

# LOAD FILES FOR PLOT ESTIMATION (temporary)
'''cov_x = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(1)+'.txt')
cov_y = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(2)+'.txt')   
cov_vx = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(3)+'.txt') 
cov_vy = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(4)+'.txt') 
x_hat_x = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(1)+'.txt')
x_hat_y = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(2)+'.txt')'''



# Downsampling script
original_samples = samples
sampling = 10
samples = int(np.ceil(original_samples/sampling))
target_x = np.zeros((samples,int(targetNum)))
target_y = np.zeros((samples,int(targetNum)))
for i in range(targetNum):
    tmp_x = target_x_traj[:,i]
    tmp_y = target_x_traj[:,i]

    target_x[:,i] = tmp_x[::sampling]
    target_y[:,i] = tmp_y[::sampling]

auv_x = np.zeros((samples,int(auvNum)))
auv_y = np.zeros((samples,int(auvNum)))

for i in range(int(auvNum)):
    tmp_x = auv_x_traj[:,i]
    auv_x[:,i] = tmp_x[::sampling]
    tmp_y = auv_y_traj[:,i]
    auv_y[:,i] = tmp_y[::sampling]
    tmp = surge_vel[i]
    surge_vel[i] = tmp[::sampling]
    tmp = heading[i]
    heading[i] = tmp[::sampling]

# Utils functions
def computeCost(phi):

    length_y = len(phi)
    W = np.zeros((length_y,length_y))
    for i in range(length_y):
        W[i,i] = 1.0
    PHI = np.dot(np.transpose(phi),np.dot(np.linalg.inv(W),phi))

    return np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)

# Print some simulation info
print('SIMULATION INFO [auvNum - Simulation Time (s) - Slot Time (s)]',sim_info)   
print('Acoustic Communication Stat [PDR AUV1,PDR AUV2,PDR AUV3,PDR AUV4]:',PDR)
#avgTimeOpt = 0.0
for i in range(int(auvNum)):
    #print('OPTIMIZATION STATS --> Average Optimization Time AUV ID:',i+1,sum(avgTime[i])/len(avgTime[i]))
    #avgTimeOpt+= sum(avgTime[i])/len(avgTime[i])
    print('AUV ID RMSE (m):',i+1,(sum(tracking_errors[i])/len(tracking_errors[i])))

#avgTimeOpt = avgTimeOpt/auvNum
###############################################
# Animated plot

# Parameters
lw = 8
fs = 50
tw = 40
tw = 20
AUV_failure = False

# Initialize data structures with predefined sizes
a_x = [[] for _ in range(auvNum)]
a_y = [[] for _ in range(auvNum)]
l_x = [[] for _ in range(auvNum)]
l_y = [[] for _ in range(auvNum)]
t_x = [[] for _ in range(targetNum)]
t_y = [[] for _ in range(targetNum)]
phi_lists = np.zeros((4, samples, 2))  # Structured as (AUVs, samples, 2 angles)

phi = np.zeros((4, 2))
list_phi = np.zeros(samples)

# Populate phi_lists based on target positions
for k in range(targetNum):
    tmp_t_x = target_x[:, k]
    tmp_t_y = target_y[:, k]

    for j in range(auvNum):
        tmp_x = auv_x[:, j]
        tmp_y = auv_y[:, j]
        
        for i in range(samples):
            angle = atan2(tmp_y[i] - tmp_t_y[i], tmp_x[i] - tmp_t_x[i])
            phi_lists[j, i] = [sin(angle), -cos(angle)]
            
            # Storing positions and line data for potential plotting or debugging
            t_x[k].append(tmp_t_x[i])
            t_y[k].append(tmp_t_y[i])
            a_x[j].append(tmp_x[i])
            a_y[j].append(tmp_y[i])
            l_x[j].append([tmp_x[i], tmp_t_x[i]])
            l_y[j].append([tmp_y[i], tmp_t_y[i]])

    # Populate phi and compute cost with conditions
    for i in range(samples):
        if header.config.AUV_failure:
            phi[:3] = phi_lists[:3, i]  #TODO better Only assign the first 3 AUVs' data
        else:
            phi[:4] = phi_lists[:4, i]  # Assign all 4 AUVs' data
        
        cost = computeCost(phi)
        list_phi[i] += cost


math_vars = ['\\xi','\\hat{\\xi}','s_i','\\kappa(\\Phi)']
math_vars = ['Target','\\hat{\\xi}','AUVs','LOSS FUNCTION']
time_vars = ['(t_{0})','(t_{f})']
sensors_vars = ['s_1','s_2','s_3']
sensors_vars = ['AUV1','AUV2','AUV3','\\epsilon']
auv_vars = ['AUV1','AUV2','AUV3','\\epsilon']

# Initialize the figure

fig = plt.figure()
fig.patch.set_facecolor('lightgray')  # Background color for the figure
# Adjust the figure size (for full screen) and define grid spec
fig.set_size_inches(16, 9)
# Set subplots spacing
#fig.tight_layout(pad=10.0)
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[:, 1:3])  # This takes the entire first row
ax2 = fig.add_subplot(gs[0, 0])  # This takes the bottom left
ax3 = fig.add_subplot(gs[1, 0])  # This takes the bottom right
ax4 = fig.add_subplot(gs[2, 0])  # This takes the bottom right

########################### PLOT SIM INFO ##############################
ax4.set_ylim([0,100])
ax4.set_xlim([0,100])
ax4.set_ylabel('Simulation Info')
math_vars2 = ['\\sigma_m']
interval = 80
ax4.text(10,78,'Simulation Time: '+str(simTime)+' (s) \n Accelerated view',fontsize=2*tw/3)
ax4.text(10,60,'Packet Delivery Ratio: 75 %',fontsize=2*tw/3)
ax4.text(10,30,'Measurament noise '+r'$ %s $'%math_vars2[0]+' = 5 (deg)\n+Outliers 20%',fontsize=2*tw/3)
ax4.text(10,20,'TDMA slot time: 4 (sec)',fontsize=2*tw/3)
#ax4.text(10,2,'Average optimization time = '+str(np.round(avgTimeOpt*5,3))+' (sec)',fontsize=2*tw/3)
#ax4.set_title('Simulation Parameters',y=-0.01)
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 


############################ PLOT TRACKING ERROR #############################
for i in range(int(auvNum)):

    t_prova = np.linspace(0,simTime,len(tracking_errors[i]))
    tmp = tracking_errors[i]
    if i == 0:
        tmp_err1 = tracking_errors[i]
        err1 = ax3.plot(t_prova[0],tmp_err1[0],label=r'$ %s $'%sensors_vars[i],linewidth=lw)[0]
    elif i == 1:
        tmp_err2 = tracking_errors[i]
        err2 = ax3.plot(t_prova[0],tmp_err2[0],label=r'$ %s $'%sensors_vars[i],linewidth=lw)[0]
    else:
        tmp_err3 = tracking_errors[i]
        err3 = ax3.plot(t_prova[0],tmp_err3[0],label=r'$ %s $'%sensors_vars[i],linewidth=lw)[0]
    max_value = tmp.max()
epsi = []    
for i in range(len(t_prova)):
    epsi.append(0.0)
err = ax3.plot(t_prova,epsi,'r--',linewidth=lw,label=r'$ %s $'%sensors_vars[-1])   

ax3.set_ylim([0,4])
ax3.set_xlabel('t (s)', labelpad=0.01)
ax3.set_ylabel('RMSE (m)')
ax3.legend(fontsize=fs/5, loc='upper right')
ax3.grid()

#################### PLOT LOSS FUNCTION #######################
tmp = np.linspace(0,simTime,samples)
t = []
for i in range(samples):
    t.append(tmp[i])

#cost_lines = ax2.plot(t[0],list_phi[0],linewidth=lw,label='LOSS FUNCTION')[0]#label=r'$ %s $'%math_vars[3])[0]
cost_lines = [ax2.plot([], [], linewidth=lw / 2,label='LOSS FUNCTION')[0]] 
opt_value = []
for i in range(samples):
    opt_value.append(1)
ax2.plot(t,opt_value,'r--',linewidth=lw,label='Optimal Value')
ax2.set_xlabel('t (s)',labelpad=0.01)
ax2.set_ylabel('Cumulative Loss Function')
ax2.set_ylim([0,list_phi.max()+1])
#ax2.set_ylabel(r'$ %s $'%math_vars[3], fontsize=fs)
ax2.legend(fontsize=fs/5,loc='upper right')
ax2.grid()

################## PLOT SIMULATION SCENARIO ###################################
# Initialize lines and scatter points for targets
target_lines = [ax1.plot([], [], 'r', linewidth=lw / 2)[0] for _ in range(targetNum)]
target_scatters = [ax1.scatter([], [], c='r', linewidths=lw) for _ in range(targetNum)]

# Initialize lines and scatter points for AUVs
auv_lines = [ax1.plot([], [], 'darkslategrey', linewidth=lw / 2)[0] for _ in range(auvNum)]
auv_scatters = [ax1.scatter([], [], c='darkslategrey', linewidths=lw) for _ in range(auvNum)]

# Initialize LOS lines
los_lines = [ax1.plot([], [], 'g--', linewidth=lw / 4)[0] for _ in range(auvNum)]

# Setting axis properties
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.grid()
ax1.axis('equal')
ax1.set_xlim([-1000, +1000])
ax1.set_ylim([-1000, +1000])

# Initialization function
def init():
    for line in auv_lines + los_lines + target_lines + cost_lines:
        line.set_data([], [])
    for scatter in auv_scatters:
        scatter.set_offsets(np.empty((0, 2)))  # Ensure empty 2D array
    return auv_lines + auv_scatters + los_lines + target_lines + cost_lines

# Update function for each frame
def update(frame):
    # Update the loss function line
    cost_lines[0].set_data(t[:frame],list_phi[:frame])

    # Update line paths for each AUV
    for i in range(auvNum):
        auv_lines[i].set_data(a_x[i][:frame], a_y[i][:frame])
        auv_scatters[i].set_offsets(np.array([[a_x[i][frame], a_y[i][frame]]]))  # Correctly shape offsets

        # Update LOS lines if available up to the current frame
        los_lines[i].set_data(l_x[i][:frame], l_y[i][:frame])
    for i in range(targetNum):
        target_lines[i].set_data(t_x[i][:frame], t_y[i][:frame])
        target_scatters[i].set_offsets(np.array([[t_x[i][frame], t_y[i][frame]]]))

    # Example condition to change colors or clear elements
    if frame >= simTime / 2 and AUV_failure:
        los_lines[1].set_data([], [])  # Clear LOS line for AUV 2 if AUV fails
        auv_scatters[1].set_facecolor('orange')  # Change color for AUV 2
    else:
        auv_scatters[1].set_facecolor('darkslategrey')

    return auv_lines + auv_scatters + los_lines + target_lines + cost_lines

# Run the animation
ani = animation.FuncAnimation(fig, update, frames=samples, init_func=init, blit=True, interval=0.1)
plt.show()

#ani.save(filename="/home/andrea/animations/realistic.mp4", writer='ffmpeg', fps=30,dpi=200)  # Increase DPI for better quality)
#ani2.save(filename="/home/andrea/animations/fixed_target_cost.gif", writer="pillow")
