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

###############################################
# Static plot

# Downsampling script
original_samples = samples
sampling = 10
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
        
        cost = header.utils.computeCost(phi)
        list_phi[i] += cost

# PLOT SETUP
# Parameters
fs = 40
lw = 5
sw = 5
tw = 30
AUV_failure = False

vars = ['Target','\\hat{\\xi}','AUVs','LOSS FUNCTION']
vars_math = ['\\kappa(\\Phi)','\\xi','C^{(d)}+C^{(g)}']

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
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  # This takes the bottom left
ax2 = fig.add_subplot(gs[1, 0])  # This takes the bottom right
ax3 = fig.add_subplot(gs[0, 1])  # This takes the bottom right
ax4 = fig.add_subplot(gs[1, 1])  # This takes the bottom right

########################### PLOT SIM INFO ##############################
ax3.set_ylim([0,100])
ax3.set_xlim([0,100])
ax3.set_ylabel('Simulation Info')
math_vars2 = ['\\sigma_m']
ax3.text(10,78,'Simulation Time: '+str(simTime)+' (s) \n Accelerated view',fontsize=2*tw/3)
ax3.text(10,60,'Packet Delivery Ratio: 75 %',fontsize=2*tw/3)
ax3.text(10,30,'Measurament noise '+r'$ %s $'%math_vars2[0]+' = 5 (deg)\n+Outliers 20%',fontsize=2*tw/3)
ax3.text(10,20,'TDMA slot time: 4 (sec)',fontsize=2*tw/3)
#ax3.text(10,2,'Average optimization time = '+str(np.round(avgTimeOpt*5,3))+' (sec)',fontsize=2*tw/3)
#ax3.set_title('Simulation Parameters',y=-0.01)
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

############################ PLOT TRACKING ERROR #############################
# Initialize lists to hold errors and plots
t_prova = [np.linspace(0, simTime, len(tracking_errors[i])) for i in range(int(auvNum))]
errors = [tracking_errors[i] for i in range(int(auvNum))]
error_plots = []

# Plot each AUV's tracking error
for i, (t, err) in enumerate(zip(t_prova, errors)):
    error_plot = ax2.plot(t, err, label=r'$ %s $' % agents_vars[i], linewidth=lw)
    error_plots.append(error_plot)
    max_value = max(max_value, err.max()) if i > 0 else err.max()  # Keep track of the max value across errors

epsi = []    
for i in range(len(t_prova[0])):
    epsi.append(0.0)
err = ax2.plot(t_prova[0],epsi,'r--',linewidth=lw,label=r'$ %s $'%agents_vars[-1])   

ax2.set_xlabel('t (s)', labelpad=0.01)
ax2.set_ylabel('RMSE (m)')
ax2.legend(fontsize=fs/5, loc='upper right')
ax2.grid()

#################### PLOT LOSS FUNCTION #######################
tmp = np.linspace(0,simTime,samples)
t = []
for i in range(samples):
    t.append(tmp[i])

ax1.plot(t, list_phi, linewidth=lw / 2,label='LOSS FUNCTION')
opt_value = []
for i in range(samples):
    opt_value.append(1)
ax1.plot(t,opt_value,'r--',linewidth=lw,label='Optimal Value')
ax1.set_xlabel('t (s)',labelpad=0.01)
ax1.set_ylabel('Cumulative Loss Function')
ax1.set_ylim([0,list_phi.max()+1])
#ax1.set_ylabel(r'$ %s $'%vars[3], fontsize=fs)
ax1.legend(fontsize=fs/5,loc='upper right')
ax1.grid()

################## PLOT SIMULATION SCENARIO ###################################
fig = plt.figure()
ax = fig.add_subplot()  # This takes the entire first row
colors = []

# Plot the AUV trajectories (a_x and a_y) with labels and different colors
a = 0
test = []
for i in range(samples):
    a = 0 + i#samples - i
    test.append(a*10)

c_map = ax.scatter(target_x[:,0],target_y[:,0],c=test,cmap='autumn_r',vmin=0, vmax=simTime,linewidths=sw)
ax.scatter(target_x[0],target_y[0],c='y',marker='o',linewidths=sw)
ax.scatter(target_x[-1],target_y[-1],c='k',linewidths=sw)
ax.scatter(target_x[-1],target_y[-1],c='r',linewidths=sw)

ax.text(target_x[-1],target_y[-1],r'$ %s $'%vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
ax.text(target_x[0],target_y[0],r'$ %s $'%vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)

cb = fig.colorbar(c_map, ax=ax)
cb.set_label('t (s)',fontsize=fs)
cb.ax.tick_params(labelsize=(fs/3)*2)

for i in range(int(auvNum)):
    
    if AUV_failure == True:
        if i != 1:
            ax.plot([auv_x[-1,i],
                target_x[-1,0]],[auv_y[-1,i],target_y[-1,0]],'r--',linewidth=lw/3)
    else:
        ax.plot([auv_x[-1,i],
            target_x[-1,0]],[auv_y[-1,i],target_y[-1,0]],'r--',linewidth=lw/3,label='LOS'+r'$ %s $'%time_vars[1])

    ax.scatter(auv_x[:,i],auv_y[:,i],c=test,cmap='autumn_r',linewidths=sw)
    ax.text(auv_x[0,i],auv_y[0,i],r'$ %s $'%agents_vars[i]+r'$ %s $'%time_vars[0],fontsize=tw)
    ax.scatter(auv_x[0,i],auv_y[0,i],c='y',linewidths=sw)
    
    if AUV_failure == True:
        if i == 1:  
            ax.text(auv_x[-1,i],auv_y[-1,i],'FAILURE',fontsize=tw)
            ax.scatter(auv_x[-1,i],auv_y[-1,i],marker='X',c='r',linewidths=sw)
    else:
        ax.scatter(auv_x[-1,i],auv_y[-1,i],c='r',linewidths=sw)

ax.set_xlabel('x (m)',fontsize=fs)
ax.set_ylabel('y (m)',fontsize=fs)
ax.grid()
ax.axis('equal')
ax.legend(fontsize=(fs*2)/3,loc='lower right')
#plt.show()

##########################################################
# Plot SNR between the AUVs given the desired topology

fig6, ax = plt.subplots()
sampling = 1
math_vars = ['SNR_{ij}(dB)','SNR_{12}','SNR_{23}','SNR_{lb}','SNR_{ub}']

f = header.config.f
acoustic_loss =  0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003 #f is in kHz

snr_12= []
snr_23 = []

tmp_x_1 = auv_x[:,0]
tmp_y_1 = auv_y[:,0]
tmp_x_2 = auv_x[:,1]
tmp_y_2 = auv_y[:,1]
tmp_x_3 = auv_x[:,2]
tmp_y_3 = auv_y[:,2]
NL = header.config.NL
loops = len(tmp_x)
for j in range(loops):
    if j == loops/2:
        NL = header.config.NL
    dist_12 = np.sqrt((tmp_x_1[j]-tmp_x_2[j])**2+(tmp_y_1[j]-tmp_y_2[j])**2)
    dist_23 = np.sqrt((tmp_x_2[j]-tmp_x_3[j])**2+(tmp_y_2[j]-tmp_y_3[j])**2)

    TL_12 = 20*np.log10(dist_12) + (dist_12*acoustic_loss*1e-3)
    TL_23 = 20*np.log10(dist_23) + (dist_23*acoustic_loss*1e-3)
                    
    snr_12.append(header.config.SL - NL - TL_12 + header.config.DI)
    snr_23.append(header.config.SL - NL - TL_23 + header.config.DI)

ax.plot(t,snr_12,label=r'$ %s $'%math_vars[1],linewidth=lw) 
ax.plot(t,snr_23,label=r'$ %s $'%math_vars[2],linewidth=lw)
SNR_lb, SNR_ub = [], []
noise_change = []
for i in range(loops):
    SNR_lb.append(header.config.SNR_lb)
    SNR_ub.append(header.config.SNR_ub)
    noise_change.append(header.config.TIME_DURATION/2)
ax.plot(t,SNR_lb,'r--',label=r'$ %s $'%math_vars[3],linewidth=lw)
ax.plot(t,SNR_ub,'g--',label=r'$ %s $'%math_vars[4],linewidth=lw)
#plt.axvline(x=header.config.TIME_DURATION/2,color='k',label='NOISE CHANGE',linewidth=lw)

ax.set_ylabel(r'$ %s $'%math_vars[0], fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.show()

