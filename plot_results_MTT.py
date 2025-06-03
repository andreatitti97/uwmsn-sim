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

save_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/logs_paper-ETC'
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
estimationETC = [[] for _ in range((auvNum))]
surge_vel = [[] for _ in range((auvNum))]
heading = [[] for _ in range(auvNum)]
x_hat, tracking_errors, P, avgNodes, avgTime = [], [], [], [], []
PDR = np.zeros(((auvNum),1))

# Load target data - TO DOWNSAMPLE
for i in range(targetNum):
    target_x_traj[:,i] = np.loadtxt(log_directory+'/target_x_traj'+str(i+1)+'.txt')
    target_y_traj[:,i] = np.loadtxt(log_directory+'/target_y_traj'+str(i+1)+'.txt')


tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_1.txt')
n_estimations = len(tmp)


for i in range(int(auvNum)):
    # AUVs Simulation Data - TO DOWNSAMPLE
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv_x_traj'+str(i+1)+'.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv_y_traj'+str(i+1)+'.txt')

    guidanceETC[i] = np.loadtxt(log_directory+'/'+str(i+1)+'etcGuidance.txt')
    estimationETC[i] = np.loadtxt(log_directory+'/'+str(i+1)+'etcEstimation.txt')
    #surge_vel[i] = np.loadtxt(log_directory+'/'+str(i+1)+'surge_vel')
    #heading[i] = np.loadtxt(log_directory+'/'+str(i+1)+'heading')*180/np.pi

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

    # Estimation Data
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-trackErr.txt')
    tracking_errors.append(err)

    x_hat_j = np.zeros((n_estimations,4))
    
    cov = np.zeros((n_estimations,4))
    #for j in range(auvNum):
    x_hat_j = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_1.txt')
    if len(x_hat_j) < n_estimations:
        n_estimations = len(x_hat_j)

        #cov[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-cov_1.txt')
    x_hat.append(x_hat_j[:n_estimations,:])
    #P.append(cov)

# Downsampling script

failed_auv = 1

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
            
            phi_lists[j, i] = [np.sin(angle), -np.cos(angle)]
            
            # Storing positions and line data for potential plotting or debugging
            t_x[k].append(tmp_t_x[i])
            t_y[k].append(tmp_t_y[i])
            a_x[j].append(tmp_x[i])
            a_y[j].append(tmp_y[i])
            l_x[j].append([tmp_x[i], tmp_t_x[i]])
            l_y[j].append([tmp_y[i], tmp_t_y[i]])#
            
    once = False
    epsi = 0.0
    # Populate phi and compute cost with conditions
    for i in range(samples):
        if header.config.AUV_failure and i >= samples / 2:
            if i >= samples / 2:
                epsi = 0.0
            if once == False:
                phi_lists = np.delete(phi_lists, (failed_auv-1), axis=0)#for now only AUV2 can fail
                once = True
            phi[:auvNum-1] = phi_lists[:, i]
        else:
            phi[:auvNum] = phi_lists[:auvNum, i]  # Assign all 4 AUVs' data
        #cost1 = header.utils.computeCost(phi[:2,:])
        #cost2 = header.utils.computeCost(phi[1:,:])

        cost = header.utils.computeCost(phi)
        
        list_phi[i] += cost+epsi#cost1 + cost2 #+ dist1[i] + dist2[i] + dist3[i]


# Print some simulation info
print('SIMULATION INFO [auvNum - Simulation Time (s) - Slot Time (s)]',sim_info)   
print('Acoustic Communication Stat [PDR AUV1,PDR AUV2,PDR AUV3,PDR AUV4]:',PDR)
avgError = []
avgVar = []
for i in range(int(auvNum)):
    #print('OPTIMIZATION STATS --> Average Optimization Time AUV ID:',i+1,sum(avgTime[i])/len(avgTime[i]))
    mean_error = (sum(tracking_errors[i])/len(tracking_errors[i]))

    variance_error = np.var(tracking_errors[i])#sum((e - mean_error) ** 2 for e in tracking_errors[i]) / len(tracking_errors[i])
    print(f'AUV ID {i+1} RMSE (m): {mean_error:.3f}, Std Dev: {np.sqrt(variance_error):.3f}')
    avgError.append(mean_error)
    avgVar.append(np.sqrt(variance_error))

#print('alphaETC=',header.config.alphaETC)
print('Slot TDMA (s)',sim_info[0])
print('AVERAGE TRACKING ERRORS:',np.sum(avgError)/len(avgError))
print('AVERAGE VARIANCR:',np.sum(avgVar)/len(avgVar))
pings = 0.0
for i in range(auvNum):
    tmp = len(guidanceETC[i])
    pings += tmp
avg_pings = pings/auvNum
horizon = 5
comm_load_bit = avg_pings*(4+2*horizon)*4*8
print('COMMUNICATION LOAD ETC GUIDANCE',comm_load_bit)
#np.savetxt('/home/andrea/Documents/controlo_paper_results/official_results/validation1/logs/logs_COMPARISON_ERRORS/cond_range3',list_phi)
#np.savetxt('/home/andrea/Documents/controlo_paper_results/official_results/validation1/logs/logs_COMPARISON_ERRORS/err_range3',tracking_errors[0])
#np.savetxt('/home/andrea/Desktop/cost_gamma=0.3',list_phi)
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
# Plot Communication info

figComm, axComm = plt.subplots(figsize=(10, 5 * auvNum-1))

Ts = 12
min_time_diff = 12  # Minimum allowed time between two pings
colors = plt.cm.get_cmap('Accent', auvNum)  # Get a colormap with a different color for each AUV

for j in range(auvNum):
    
    if guidanceETC[j] is not None:
        guidanceETC[j] = guidanceETC[j].tolist()  # Convert numpy array to list
        for i in range(len(guidanceETC[j])):
            guidanceETC[j][i] = int(guidanceETC[j][i])
        
        # Ensure each AUV has a ping at j * Ts
        guidanceETC[j].append(j * Ts)
        guidanceETC[j] = sorted(guidanceETC[j])
        
        # Filter out pings that are too close to each other
        filtered_pings = [guidanceETC[j][0]]
        for t in guidanceETC[j][1:]:
            if t - filtered_pings[-1] >= min_time_diff:
                filtered_pings.append(t)

        pingAUVj = [j + 1] * len(filtered_pings)  # Offset each AUV's y-value by its index
        axComm.scatter(filtered_pings, pingAUVj, label=f'ping AUV {j+1}', color=colors(j), marker='X', s=500, linewidths=1)

axComm.set_yticks(np.linspace(1, auvNum, auvNum))  # Increase horizontal grid lines
axComm.set_title('Communication Info: ETC guidance routins', fontsize=fs)
axComm.set_xlabel('t(s)', fontsize=fs)
axComm.set_ylabel('AUV ID', fontsize=fs)
#axComm.legend(fontsize=fs * 3 / 5)
axComm.grid()
axComm.tick_params(axis='both', which='major', labelsize=ticks_size)

###########################################################
# Plot Communication info

figComm, axComm = plt.subplots(figsize=(10, 5 * auvNum-1))

Ts = 12
min_time_diff = 12  # Minimum allowed time between two pings
colors = plt.cm.get_cmap('Accent', auvNum)  # Get a colormap with a different color for each AUV

for j in range(auvNum):
    
    if estimationETC[j] is not None:
        estimationETC[j] = estimationETC[j].tolist()  # Convert numpy array to list
        for i in range(len(estimationETC[j])):
            estimationETC[j][i] = int(estimationETC[j][i])
        
        # Ensure each AUV has a ping at j * Ts
        estimationETC[j].append(j * Ts)
        estimationETC[j] = sorted(estimationETC[j])
        
        # Filter out pings that are too close to each other
        filtered_pings = [estimationETC[j][0]]
        for t in estimationETC[j][1:]:
            if t - filtered_pings[-1] >= min_time_diff:
                filtered_pings.append(t)

        pingAUVj = [j + 1] * len(filtered_pings)  # Offset each AUV's y-value by its index
        axComm.scatter(filtered_pings, pingAUVj, label=f'ping AUV {j+1}', color=colors(j), marker='X', s=500, linewidths=1)

axComm.set_yticks(np.linspace(1, auvNum, auvNum))  # Increase horizontal grid lines
axComm.set_title('Communication Info: ETC estimation routins', fontsize=fs)
axComm.set_xlabel('t(s)', fontsize=fs)
axComm.set_ylabel('AUV ID', fontsize=fs)
#axComm.legend(fontsize=fs * 3 / 5)
axComm.grid()
axComm.tick_params(axis='both', which='major', labelsize=ticks_size)

##########################################################
# Plot trend conditioning estimation problem
fig, ax = plt.subplots()
t_axis = np.linspace(0,simTime,len(list_phi))
max_ = max(list_phi)
tmp = []
for i in range(samples):
    tmp.append(list_phi[i]/max_)

ax.plot(t_axis,list_phi,label='Loss Function',linewidth=lw)
opt_value = []
for i in range(samples):
    opt_value.append(1)
ax.plot(t_axis,opt_value,'r:',label='Optimal Value',linewidth=2)
ax.set_xlabel('t (s)', fontsize = fs)
#ax.set_ylabel(r'$ %s $'%math_vars[2], fontsize=fs)
ax.set_ylabel('Cumulative Objective', fontsize=fs)
ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=ticks_size, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=ticks_size, rotation=0)#to set dimension and orientation of tick labels
#plt.show()
np.savetxt(save_directory+'/k_phi-alpha='+str(header.config.alphaETC),list_phi)
##########################################################
# Plot tracking error in separate subplots

  # Index of the failed AUV (e.g., AUV2 fails, which is index 1)
FAILURE = header.config.AUV_failure
# Determine active AUVs
if FAILURE:
    active_auvs = [i for i in range(auvNum) if i != failed_auv-1]
else:
    active_auvs = list(range(auvNum))

# Number of active AUVs
active_auvNum = len(active_auvs)

fig, axes = plt.subplots(active_auvNum, 1, figsize=(8, 3 * active_auvNum), sharex=True)
#fig.subplots_adjust(left=0.12, right=0.14, top=0.90, bottom=0.88)
math_vars = ['AUV1', 'AUV2', 'AUV3', '\\epsilon']
t = np.linspace(0, simTime, n_estimations)
epsi = np.zeros_like(t) + 15.0  # 15 meter error - reference line

max_ = max(max(tracking_errors[i]) for i in active_auvs)

num_y_ticks = 4  # Define how many horizontal grid lines you want

# Ensure axes is iterable when there's only one subplot
if active_auvNum == 1:
    axes = [axes]

for idx, i in enumerate(active_auvs):  # Iterate over active AUV indices
    axes[idx].plot(t, tracking_errors[i][:n_estimations], label=rf'$ {math_vars[i]} $', 
                   linewidth=lw, marker='o', markersize=ms, color=colors(i))

    axes[idx].plot(t, epsi, 'r:', linewidth=lw * 2 / 3, label=rf'$ {math_vars[-1]} $')
    
    if idx == 1 or (active_auvNum == 1 and idx == 0):
        axes[idx].set_ylabel('            RMSE (m)', fontsize=42)#12 spaces for auv failure
    
    axes[idx].legend(fontsize=25)
    axes[idx].grid()
    axes[idx].tick_params(axis='y', labelsize=fs * 2 / 3)
    axes[idx].tick_params(axis='x', labelsize=fs * 2 / 3)
    
axes[-1].set_xlabel('t (s)', fontsize=42)
plt.xticks(fontsize=fs * 2 / 3)



##########################################################
# Plot consensus dynamic derivative
fig, ax = plt.subplots()
math_vars = ['\\dot{\\xi}(t)']
consensus_dyn = []
t = np.linspace(0, simTime, n_estimations)
colors = plt.cm.get_cmap('tab10', auvNum)  # Get a colormap with a different color for each AUV

x_hat1 = x_hat[0]
x_hat2 = x_hat[1]
x_hat3 = x_hat[2]
epsi = 1.5
epsi_max = np.zeros_like(t)+epsi
epsi_min = np.zeros_like(t)-epsi

# Compute the consensus dynamic as the sum of pairwise differences (norms)
for i in range(n_estimations):
    diff1 = np.linalg.norm(x_hat1[i] - x_hat2[i])
    diff2 = np.linalg.norm(x_hat1[i] - x_hat3[i])
    diff3 = np.linalg.norm(x_hat2[i] - x_hat3[i])
    consensus_dyn.append(diff1 + diff2 + diff3)


# Compute the numerical derivative (first derivative) of the consensus dynamic
consensus_derivative = np.gradient(consensus_dyn, t)

ax.plot(t, consensus_derivative, linewidth=lw, marker='o', markersize=ms, label=r'$ %s $' % math_vars[0])
#ax.plot(t, consensus_derivative, linewidth=lw, marker='o', label=r'$ %s $' % math_vars[0])
ax.plot(t, epsi_min, 'r:', linewidth=lw, label=r'$ -\epsilon $')
ax.plot(t, epsi_max, 'r:', linewidth=lw, label=r'$ +\epsilon $')
ax.set_xlabel('t (s)', fontsize=fs)
ax.set_ylabel(r'$ %s $' % math_vars[0], fontsize=fs)
ax.legend(fontsize=ticks_size/2)
ax.grid()
plt.yticks(fontsize=ticks_size, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=ticks_size, rotation=0)#to set dimension and orientation of tick labels

##########################################################
# Plot simulation scenario

# PLOT SETUP
fs = 42
lw = 5
sw = 1
tw = 40
lwsw = 150
ms = 2
math_vars = ['\\kappa(\\Phi)','\\xi','C^{(d)}+C^{(g)}']
agents_vars = ['s_1','s_2','s_3','s_4']
time_vars = ['(t_{0})','(t_{f})','(t_{0}=t_{f})']

fig2, ax = plt.subplots()
#fig2.subplots_adjust(left=0.12, right=0.14, top=0.90, bottom=0.88)
#ax = fig2.add_axes([0.2, 0.2, 0.88, 0.88])  # [left, bottom, width, height] in figure coordinates

plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels

test = np.linspace(0, simTime, samples)  # Create a test array for color mapping

c_map = ax.scatter(target_x[:, 0], target_y[:, 0], c=test, cmap='autumn_r', vmin=0, vmax=simTime, linewidths=sw)

ax.scatter(target_x[0],target_y[0],c='y',marker='X',s=lwsw,linewidths=ms,label=r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0])
#ax.scatter(target_x[0],target_y[0],c='k',linewidths=sw*12)
ax.scatter(target_x[-1],target_y[-1],c='r',marker='X',s=lwsw,linewidths=ms,label=r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1])
#ax.scatter(target_x[-1],target_y[-1],c='r',linewidths=sw*8)


#ax.text(target_x[-1],target_y[-1]-23,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)
#ax.text(target_x[-1]-200,target_y[-1],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
#ax.text(target_x[0]-150,target_y[0],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)

#ax.text(target_x[-1]+5,target_y[-1]+2,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)

cb = fig.colorbar(c_map, ax=ax)
cb.set_label('t (s)',fontsize=fs)
cb.ax.tick_params(labelsize=(fs/3)*2)


for i in range(int(auvNum)):
    
    if i == 1000:
        ax.plot([auv_x[-1,i],
            target_x[-1,0]],[auv_y[-1,i],target_y[-1,0]],'r:',linewidth=lw/3,label='LOS'+r'$ %s $'%time_vars[1])
    else:
        if i != failed_auv-1 or FAILURE == False:
            ax.plot([auv_x[-1,i],
                target_x[-1,0]],[auv_y[-1,i],target_y[-1,0]],'r:',linewidth=lw/3)
        
    if i == 0:
        a,b = -50,-35
    elif i == 1:
        a,b = -50,25
    else:
        a,b = -60,+25

    '''if i == 0:
        a,b,c,d = +5,+0,+2,0
    elif i == 1:
        a,b,c,d = +5,+0,-0,+5
    else:
        a,b,c,d = -15,-8,-3,+5'''

    ax.scatter(auv_x[:,i],auv_y[:,i],c=test,cmap='autumn_r',linewidths=sw)
    ax.text(auv_x[0,i]+a,auv_y[0,i]+b,r'$ %s $'%agents_vars[i]+r'$ %s $'%time_vars[0],fontsize=tw)
    ax.scatter(auv_x[0,i],auv_y[0,i],c='y',linewidths=sw*8)
    #ax.text(auv_x[-1,i]+c,auv_y[-1,i]+d,r'$ %s $'%agents_vars[i]+r'$ %s $'%time_vars[1],fontsize=tw)
    if i == failed_auv-1 and FAILURE==True:  
        
        ax.text(auv_x[-1,i]-55,auv_y[-1,i]+20,'FAILURE',fontsize=3*tw/5)
        ax.scatter(auv_x[-1,i],auv_y[-1,i],marker='X',c='r',linewidths=sw*15)
    else:   
        ax.scatter(auv_x[-1,i],auv_y[-1,i],c='r',linewidths=sw*8)


'''PROPOSAL
ax.text(target_x[-1],target_y[-1]+5,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)
if i == 0:
        a,b,c,d = +5,+0,+2,0
    elif i == 1:
        a,b,c,d = +5,+0,-0,+5
    else:
        a,b,c,d = -5,-8,-3,+5'''

'''RANGE

ax.text(target_x[-1]-5,target_y[-1]+4,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)
if i == 0:
        a,b,c,d = +5,-10,-10,-10
    elif i == 1:
        a,b,c,d = +5,+0,-3,+8
    else:
        a,b,c,d = -10,-8,0,-7'''

'''GEOM
ax.text(target_x[-1]+6,target_y[-1]-18,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
ax.text(target_x[0]+15,target_y[0],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)
if i == 0:
        a,b,c,d = +5,0,0,+12
    elif i == 1:
        a,b,c,d = 0,+10,-50,-5
    else:
        a,b,c,d = +15,0,0,+5'''


'''REALISTIC
ax.text(target_x[-1]+6,target_y[-1]-18,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
ax.text(target_x[0]+15,target_y[0],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)


if i == 0:
        a,b,c,d = +5,+0,+10,0
    elif i == 1:
        a,b,c,d = -45,+13,-0,-35
    else:
        a,b,c,d = +15,-8,-30,+30
'''

ax.set_xlabel('x (m)',fontsize=fs)
ax.set_ylabel('y (m)',fontsize=fs)
ax.grid()
ax.axis('equal')
ax.legend(fontsize=(fs*2)/3)

plt.show()

##########################################################
#plot distance to target
fig5, ax = plt.subplots()
sampling = 1
math_vars = ['d_1','d_2','d_3','d_4','d_{-}^{r}','d_{+}^{r}']
lthres = header.config.RANGE_TO_TARGET
x = np.linspace(0,simTime,int(samples/sampling)) #subsampled set
for i in range(int(auvNum)):
    #plt.subplot(int(auvNum),1,i+1)
    dist = []
    low_thresh = []
    tmp_x = auv_x[:,i]
    tmp_y = auv_y[:,i]
    
    for j in range(samples):

        low_thresh.append(lthres)
        dist.append(np.sqrt((target_x[j]-tmp_x[j])**2+(target_y[j]-tmp_y[j])**2))
        
        
    model = make_interp_spline(x, dist[::sampling])
    t = np.linspace(0,simTime,samples)#original samples length but interpolated
    y = model(t)
    
    ax.plot(t,y,linewidth=lw,label=r'$ %s $'%math_vars[i])

ax.set_xlabel('t (s)',fontsize=fs)
ax.set_ylabel('d (m)',fontsize=fs)
ax.plot(t,low_thresh,'r:',linewidth=lw/2,label=r'$ %s $'%math_vars[4])
ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels

########################################################################
# Plot INDIVIDUAL COST FUNCTIONS

fig5, ax = plt.subplots()
sampling = 1
math_vars = ['loss function 1','loss function 2','loss function 3']
lthres = header.config.RANGE_TO_TARGET

x = np.linspace(0,simTime,int(samples/sampling)) #subsampled set
for i in range(int(auvNum)):
    #plt.subplot(int(auvNum),1,i+1)
    reward_func = []
    dist_reward = []
    tmp_x = auv_x[:,i]
    tmp_y = auv_y[:,i]
    d0 = np.sqrt((target_x[0]-tmp_x[0])**2+(target_y[0]-tmp_y[0])**2)
    cost_d = np.linspace(5,d0,samples)
    cost_g = np.linspace(0.01,1,samples)
  
    for j in range(samples):
        tmp_phi = list_phi[j]+20

        dist = np.sqrt((target_x[j]-tmp_x[j])**2+(target_y[j]-tmp_y[j])**2)
        reward_func.append(dist+tmp_phi)
        dist_reward.append(dist)
        
        
    model = make_interp_spline(x, reward_func[::sampling])
    t = np.linspace(0,simTime,samples)#original samples length but interpolated
    y = model(t)
    
    ax.plot(t,y,linewidth=lw,label=r'$ %s $'%math_vars[i])

ax.set_xlabel('t (s)',fontsize=fs)
ax.set_ylabel('d (m)',fontsize=fs)

ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels















'''fig5, ax = plt.subplots()
ax.plot(list_phi,dist_reward,linewidth=lw/2)
ax.plot(1/cost_g,cost_d,linewidth=lw/2)
ax.scatter(list_phi[-1],dist_reward[-1],linewidths=lw/2)
ax.grid()
ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels

plt.show()'''


'''Interpolation script
x = np.linspace(0,simTime,int(len(target_x)/500)+1) #subsampled set
    
    model = make_interp_spline(x, dist[::500])

    t = np.linspace(0,simTime,len(target_x))#original samples length but interpolated
    y = model(t)
    '''

'''MOdify Ticks scripts
if i+1 < auvNum:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    else:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=True,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
'''

'''
##########################################################
# Plot distances between the AUVs
#fig6, (ax1, ax2, ax3) = plt.subplots(int(auvNum),1)
fig6, ax = plt.subplots()
sampling = 1
math_vars = ['d_{ij}','d_{12}','d_{23}','d_{31}']
for i in range(int(auvNum)):
    
    dist_ij1= []
    dist_ij2 = []
    
    if i == 0:
        idx1 = 1
        idx2 = 2
    elif i == 1:
        idx1 = 0
        idx2 = 2
    else:
        idx1 = 0
        idx2 = 1
    tmp_x_i = auv_x[:,i]
    tmp_y_i = auv_y[:,i]
    tmp_x_j1 = auv_x[:,idx1]
    tmp_y_j1 = auv_y[:,idx1]
    tmp_x_j2 = auv_x[:,idx2]
    tmp_y_j2 = auv_y[:,idx2]
    loops = len(tmp_x)
    for j in range(loops):
        dist_ij1.append(np.sqrt((tmp_x_i[j]-tmp_x_j1[j])**2+(tmp_y_i[j]-tmp_y_j1[j])**2))
        dist_ij2.append(np.sqrt((tmp_x_i[j]-tmp_x_j2[j])**2+(tmp_y_i[j]-tmp_y_j2[j])**2))
    if i == 0:
        ax.plot(t,dist_ij1,label=r'$ %s $'%math_vars[1],linewidth=lw) 
    if i == 1:
        ax.plot(t,dist_ij2,label=r'$ %s $'%math_vars[2],linewidth=lw)
    if i == 0:
        ax.plot(t,dist_ij2,label=r'$ %s $'%math_vars[3],linewidth=lw) 
 

ax.set_ylabel(r'$ %s $'%math_vars[0], fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels
#plt.show()'''

'''##########################################################
# Plot Surge Velocities

sampling = 1
fig3, ax = plt.subplots()

for i in range(int(auvNum)):

    tmp = surge_vel[i]

    x = np.linspace(0,simTime,int(len(tmp)/sampling)) #subsampled set
    model = make_interp_spline(x, tmp[::sampling])
    t = np.linspace(0,simTime,samples)#original samples length but interpolated
    y = model(t)

    ax.plot(t,y,label='AUV'+str(i+1),linewidth=lw)
    
ax.set_ylabel('u (m/s)', fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels

'''

'''##########################################################
# Plot heading 
fig4, ax = plt.subplots()
theta_var = ['\\theta_{s}']
sampling = 1
for i in range(int(auvNum)):
    tmp = heading[i]
    for j in range(len(tmp)):
        if tmp[j] < 0:
            tmp[j] = 360 + tmp[j]      

    x = np.linspace(0,simTime,int(len(tmp)/sampling)) #subsampled set

    model = make_interp_spline(x, tmp[::sampling]) #TODO CHECK ROUNDING UP PROBLEM FOR INTERP
    t = np.linspace(0,simTime,samples)#original samples length but interpolated
    y = model(t)

    ax.plot(t,y,label='AUV'+str(i+1),linewidth=lw)

ax.set_ylabel(r'$ %s $'%theta_var[0]+'(rad)', fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels
plt.show()
'''

##########################################################
# Plot SNR between the AUVs given the desired topology

'''fig6, ax = plt.subplots()
sampling = 1
math_vars = ['snr_{ij}(dB)','snr_{12}','snr_{23}','thresh snr']

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
NL = 0#header.config.NL
loops = len(tmp_x)
for j in range(loops):
    if j == loops/2:
        NL = 0#80
    dist_12 = np.sqrt((tmp_x_1[j]-tmp_x_2[j])**2+(tmp_y_1[j]-tmp_y_2[j])**2)
    dist_23 = np.sqrt((tmp_x_2[j]-tmp_x_3[j])**2+(tmp_y_2[j]-tmp_y_3[j])**2)

    TL_12 = 20*np.log(dist_12) + (dist_12*acoustic_loss*1e-3)
    TL_23 = 20*np.log(dist_23) + (dist_23*acoustic_loss*1e-3)
                    
    snr_12.append(header.config.SL - NL - TL_12 + header.config.DI)
    snr_23.append(header.config.SL - NL - TL_23 + header.config.DI)

ax.plot(t,snr_12,label=r'$ %s $'%math_vars[1],linewidth=lw) 
ax.plot(t,snr_23,label=r'$ %s $'%math_vars[2],linewidth=lw)
desired_snr = []
noise_change = []
for i in range(loops):
    desired_snr.append(header.config.DThresh)
    noise_change.append(header.config.TIME_DURATION/2)
ax.plot(t,desired_snr,'r:',label=r'$ %s $'%math_vars[2],linewidth=lw)
plt.axvline(x=header.config.TIME_DURATION/2,color='k',label='NOISE CHANGE',linewidth=lw)

ax.set_ylabel(r'$ %s $'%math_vars[0], fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels

'''