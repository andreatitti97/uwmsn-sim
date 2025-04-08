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

    #surge_vel[i] = np.loadtxt(log_directory+'/'+str(i+1)+'surge_vel.txt')
    #heading[i] = np.loadtxt(log_directory+'/'+str(i+1)+'heading.txt')*180/np.pi

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

    # Estimation Data
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-trackErr.txt')
    tracking_errors.append(err)
    #TODO Add the possibility of monitoring all the estimations
    '''tmp = np.loadtxt(log_directory+'/'+str(2)+'-x_hat_'+str(1)+'.txt')
    x_hat_ = np.zeros((len(tmp),4))
    cov = np.zeros((len(tmp),4))
    for j in range(auvNum):
        for k in range(targetNum):
            x_hat_[:,j] = np.loadtxt(log_directory+'/'+str(j)+'-x_hat_'+str(k)+'.txt')
            cov[:,j] = np.loadtxt(log_directory+'/'+str(j)+'-x_hat_'+str(k)+'.txt')
    x_hat.append(x_hat_)
    P.append(cov)'''
    

# LOAD FILES FOR PLOT ESTIMATION (temporary)
#cov_x = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(1)+'.txt')
#cov_y = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(2)+'.txt')   
#cov_vx = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(3)+'.txt') 
#cov_vy = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(4)+'.txt') 

'''x_hat = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(1)+'.txt')
x_hat_y = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(1)+'.txt')
x_hat_x = x_hat[:,0]
x_hat_y = x_hat[:,1]'''
#print(x_hat_x[0])
###############################################
# Animated plot

# Parameters
lw = 8
fs = 50
tw = 40
tw = 20
AUV_failure = False

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

# Populate phi_lists based on target positions
for k in range(targetNum):
    tmp_t_x = target_x[:, k]
    tmp_t_y = target_y[:, k]

    for j in range(auvNum):
        tmp_x = auv_x[:, j]
        tmp_y = auv_y[:, j]
        
        for i in range(samples):
            angle = atan2(tmp_y[i] - tmp_t_y[i], tmp_x[i] - tmp_t_x[i])
            if j == 0:
                dist1.append(np.sqrt((tmp_y[i] - tmp_t_y[i])**2+(tmp_x[i] - tmp_t_x[i])**2))
            elif j == 1:
                dist2.append(np.sqrt((tmp_y[i] - tmp_t_y[i])**2+(tmp_x[i] - tmp_t_x[i])**2))
            else:
                dist3.append(np.sqrt((tmp_y[i] - tmp_t_y[i])**2+(tmp_x[i] - tmp_t_x[i])**2))
            
            phi_lists[j, i] = [sin(angle), -cos(angle)]
            
            # Storing positions and line data for potential plotting or debugging
            t_x[k].append(tmp_t_x[i])
            t_y[k].append(tmp_t_y[i])
            a_x[j].append(tmp_x[i])
            a_y[j].append(tmp_y[i])
            l_x[j].append([tmp_x[i], tmp_t_x[i]])
            l_y[j].append([tmp_y[i], tmp_t_y[i]])
            
    once = False
    # Populate phi and compute cost with conditions
    for i in range(samples):
        if header.config.AUV_failure and i >= samples / 2:
            if once == False:
                phi_lists = np.delete(phi_lists, (0), axis=0)#for now only AUV2 can fail
                once = True
            phi[:auvNum-1] = phi_lists[:, i]
        else:
            phi[:auvNum] = phi_lists[:auvNum, i]  # Assign all 4 AUVs' data
        cost = header.utils.computeCost(phi)
        list_phi[i] += cost#cost1 + cost2 #+ dist1[i] + dist2[i] + dist3[i]

# Initialize strings fo legends
vars = ['Target','\\hat{\\xi}','AUVs','LOSS FUNCTION']
vars_math = ['\\kappa(\\Phi)','\\xi','C^{(d)}+C^{(g)}','\\sigma_m','\\epsilon']

time_vars = ['(t_{0})','(t_{f})','(t_{0}=t_{f})']

targets_vars = ['T'+str(i+1) for i in range(targetNum)]
targets_vars_math = ['\\xi_'+str(i+1) for i in range(targetNum)]

agents_vars = ['AUV'+str(i+1) for i in range(auvNum)]
agents_vars_math = ['s_'+str(i+1) for i in range(auvNum)]
colors = ['darkslategrey', 'orange', 'purple', 'blue']


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

ax4.text(10,78,'Simulation Time: '+str(simTime)+' (s) \n Accelerated view',fontsize=2*tw/3)
ax4.text(10,60,'Packet Delivery Ratio: '+str(header.config.PDR-5)+' %',fontsize=2*tw/3)
ax4.text(10,30,'Measurament noise '+r'$ %s $'%vars_math[3]+' = '+str(np.ceil(header.config.SIGMA_MEAS*180/np.pi))+' (deg)\n+Outliers 0%',fontsize=2*tw/3)
ax4.text(10,20,'TDMA slot time: '+str(header.config.Ts)+' (sec)',fontsize=2*tw/3)
#ax4.text(10,2,'Average optimization time = '+str(np.round(avgTimeOpt*5,3))+' (sec)',fontsize=2*tw/3)
#ax4.set_title('Simulation Parameters',y=-0.01)
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 


############################ PLOT TRACKING ERROR #############################

def generate_sequence(start, end, width):
    """
    Generate a sequence of values from start to end with a given step width.
    Ensures the last value is included if possible.

    :param start: Initial value
    :param end: Final value
    :param width: Step width
    :return: List of evenly spaced values
    """
    return np.arange(start, end, width).tolist()

# Initialize lists to hold plots
error_plots = []
initial_ping = [0,0,0]
pings = []
for i in range(int(auvNum)):
    pings.append([])
    pings[i] = generate_sequence(initial_ping[i],simTime,header.config.Ts*auvNum)

t_prova = [np.linspace(initial_ping[i], simTime, len(tracking_errors[i])) for i in range(int(auvNum))]
epsi = np.zeros(len(t_prova[0]))  # Single list of zeros for epsi across all AUVs
error_plots = [ax3.plot([], [], linewidth=lw / 3, label=agents_vars[i],marker='x',mew=lw)[0] for i in range(auvNum)]
error_pings = [ax3.scatter([], [], linewidth=lw, label=agents_vars[i],marker='x') for i in range(auvNum)]

# Loop through each AUV to initialize error plots
for i, (t_err, err) in enumerate(zip(t_prova, tracking_errors)):

    # Track max value across errors for any further processing
    max_value = max(max_value, err.max()) if i > 0 else err.max()

# Plot a static "epsi" reference line once, outside the loop
epsi_line = ax3.plot(t_prova[0], epsi, 'r--', linewidth=lw, label=r'$ %s $' % vars_math[-1])   

ax3.set_ylim([0,max_value+1])
ax3.set_ylim([0,150])
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
from matplotlib.patches import Circle

# Initialize lines and scatter points for targets
target_lines = [ax1.plot([], [], 'r', linewidth=lw / 2)[0] for _ in range(targetNum)]
target_scatters = [ax1.scatter([], [], c='r', linewidths=lw) for _ in range(targetNum)]

# Initialiaze estimation params
est_scatters = [ax1.scatter([], [], c='azure',edgecolors='y', linewidths=lw) for _ in range(auvNum)]
est_scatters = [ax1.scatter([],[],c='azure',edgecolors='y',linewidths=lw) for _ in range(auvNum)]

# Initialize lines and scatter points for AUVs
auv_lines = [ax1.plot([], [], 'darkslategrey', linewidth=lw / 2)[0] for _ in range(auvNum)]
auv_scatters = [ax1.scatter([], [], c='darkslategrey', linewidths=lw) for _ in range(auvNum)]
init_auv = [ax1.text(a_x[i][0], a_y[i][0],r'$ %s $' % agents_vars[i]) 
            for i in range(auvNum)]

# Initialize LOS lines
los_lines = [ax1.plot([], [], 'g--', linewidth=lw / 4)[0] for _ in range(auvNum)]

# Initialize AUV circles
radius = header.config.min_distance/2
auv_circles = [Circle((0, 0), radius, color='blue', alpha=0.3) for _ in range(auvNum)]
for circle in auv_circles:
    ax1.add_patch(circle)

# Setting axis properties
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.grid()
ax1.legend()
ax1.axis('equal')
ax1.set_xlim([-600, +600])
ax1.set_ylim([-600, +600])

# Initialization function
def init():
    for line in auv_lines + los_lines + target_lines + cost_lines + error_plots:
        line.set_data([], [])
    for scatter in auv_scatters + target_scatters +  est_scatters + error_pings:
        scatter.set_offsets(np.empty((0, 2)))  # Ensure empty 2D array
    for circle in auv_circles:
        circle.set_center((0, 0))  # Reset circle positions
    return est_scatters + auv_lines + auv_scatters + los_lines + target_lines + cost_lines + error_pings + error_plots + auv_circles

# Update function for each frame
normFrame = [[] for _ in range(auvNum)]
normFrame2 = [[] for _ in range(auvNum)]
next_pings = pings
print(len(pings[0]))
def update(frame):
    # Normalize frame to total samples (for synchronous evolution)
    for i in range(auvNum):
        normFrame[i] = (int(frame / samples * len(tracking_errors[i])))
        normFrame2[i] = (int(frame / samples*len(pings[i])))
    #print('frame',frame)
    #print('normaFrame',normFrame)
    #print('normaFrame2',normFrame2)
    # Update the loss function line
    cost_lines[0].set_data(t[:frame], list_phi[:frame])

    # Update line paths for each AUV
    for i in range(auvNum):
        auv_lines[i].set_data(a_x[i][:frame], a_y[i][:frame])
        auv_scatters[i].set_offsets(np.array([[a_x[i][frame], a_y[i][frame]]]))  # Correctly shape offsets

        # Update the circle position
        auv_circles[i].set_center((a_x[i][frame], a_y[i][frame]))

        # Update LOS lines if available up to the current frame
        los_lines[i].set_data(l_x[i][frame], l_y[i][frame])

        # Update tracking error
        error_plots[i].set_data(t_prova[i][:normFrame[i]], tracking_errors[i][:normFrame[i]])
        #error_pings[i].set_offsets(np.array([[t_prova[i][normFrame[i]], tracking_errors[i][normFrame[i]]]]))
        
        tmp = next_pings[i]
        
        
        '''if (int(np.floor(frame)) == tmp[0] or frame == 0) and len(pings[i])>=normFrame[i]:
            if i == 0:
                print('FRAME',int(np.floor(frame)))
                print('PING', tmp[0])
            error_pings[i].set_offsets(np.array([[pings[i][normFrame[i]], tracking_errors[i][normFrame[i]]]]))
            next_pings[i].pop(0)'''

    for i in range(targetNum):
        target_lines[i].set_data(t_x[i][:frame], t_y[i][:frame])
        target_scatters[i].set_offsets(np.array([[t_x[i][frame], t_y[i][frame]]]))

    # Update the estimation
    #array = np.array([x_hat_x[normFrame[0]], x_hat_y[normFrame[0]]])
    #est_scatters[0].set_offsets(array)

    # Example condition to change colors or clear elements
    if frame >= simTime / 2 and AUV_failure:
        los_lines[1].set_data([], [])  # Clear LOS line for AUV 2 if AUV fails
        auv_scatters[1].set_facecolor('orange')  # Change color for AUV 2
    else:
        auv_scatters[1].set_facecolor('darkslategrey')

    return est_scatters + auv_lines + auv_scatters + los_lines + target_lines + cost_lines + error_pings + error_plots + auv_circles

# Run the animation
ani = animation.FuncAnimation(fig, update, frames=samples, init_func=init, blit=True, interval=1)
plt.show()

# ani.save(filename="/home/andrea/animations/realistic.mp4", writer='ffmpeg', fps=30, dpi=200)  # Increase DPI for better quality
# ani2.save(filename="/home/andrea/animations/fixed_target_cost.gif", writer="pillow")
