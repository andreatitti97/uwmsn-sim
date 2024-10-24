#!+usr+bin+env python
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
import os, pathlib
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

# Environment initialization
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/uwmsn-sim'
log_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/logs_I-RIM_poster'#/logs_FIXED_FORM'
class_directory = pkg_directory+'/src'+'/Classes'

# Load simulation info
sim_info = np.loadtxt(log_directory+'/sim_info.txt')
ctrl_set = np.loadtxt(log_directory+'/ctrl_set.txt')
auvNum = sim_info[0] 
elapsed_t = sim_info[1]
Ts = sim_info[2]

# Load target data - TO DOWNSAMPLE
target_x_traj = np.loadtxt(log_directory+'/target_x_traj.txt')
target_y_traj = np.loadtxt(log_directory+'/target_y_traj.txt')

# Initialize data structures for AUVs data
auv_x_traj = np.zeros((len(target_x_traj),int(auvNum)))
auv_y_traj = np.zeros((len(target_y_traj),int(auvNum)))
surge_vel = [[],[],[],[]]
heading = [[],[],[],[]]
x_hat, tracking_errors, P, avgNodes, avgTime = [], [], [], [], []
PDR = np.zeros((int(auvNum),1))

for i in range(int(auvNum)):
    # AUVs Simulation Data - TO DOWNSAMPLE
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_x_traj.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_y_traj.txt')

    surge_vel[i] = np.loadtxt(log_directory+'/'+str(i+1)+'surge_vel')
    heading[i] = np.loadtxt(log_directory+'/'+str(i+1)+'heading')*180/np.pi

    # Optimization Data
    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

    # Estimation Data
    tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_2.txt')
    x_hat_ = np.zeros((len(tmp),4))
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-err.txt')
    cov = np.zeros((len(tmp),4))
    for j in range(4):
        x_hat_[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_'+str(j+1)+'.txt')
        cov[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-cov'+str(j+1)+'.txt')
    x_hat.append(x_hat_)
    P.append(cov)
    tracking_errors.append(err)

    #Optimization Data
    #PDR[i] = np.loadtxt(log_directory+'/'+str(i+1)+'-PDR')

# LOAD FILES FOR PLOT ESTIMATION (temporary)
cov_x = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(1)+'.txt')
cov_y = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(2)+'.txt')   
cov_vx = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(3)+'.txt') 
cov_vy = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(4)+'.txt') 
x_hat_x = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(1)+'.txt')
x_hat_y = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(2)+'.txt')


# Downsampling script
sampling = 100
tmp_x = target_x_traj
target_x_traj = tmp_x[::sampling]
tmp_y = target_y_traj
target_y_traj = tmp_y[::sampling]
samples = len(target_x_traj)
est_samples = len(x_hat_x)

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
avgTimeOpt = 0.0
for i in range(int(auvNum)):
    print('OPTIMIZATION STATS --> Average Optimization Time AUV ID:',i+1,sum(avgTime[i])/len(avgTime[i]))
    avgTimeOpt+= sum(avgTime[i])/len(avgTime[i])
    print('AUV ID RMSE (m):',i+1,(sum(tracking_errors[i])/len(tracking_errors[i])))

avgTimeOpt = avgTimeOpt/auvNum
###############################################
# Animated plot

# Parameters
lw = 8
fs = 50
tw = 40
tw = 20
AUV_failure = False
# Initialize data structures
a1_x, a2_x, a3_x, a4_x = [], [], [], []
a1_y, a2_y, a3_y, a4_y = [], [], [], []

l1_x, l2_x, l3_x, l4_x = [], [], [], []
l1_y, l2_y, l3_y, l4_y = [], [], [], []

phi = np.zeros((4,2))
list_phi, phi1, phi2, phi3, phi4 = [], [], [], [], []

for j in range(int(auvNum)):
    tmp_x = auv_x[:,j]
    tmp_y = auv_y[:,j]

    for i in range(samples):
        b = atan2(tmp_y[i]-target_y_traj[i],tmp_x[i]-target_x_traj[i])

        if j == 0:
            a1_x.append(tmp_x[i])
            a1_y.append(tmp_y[i])
            l1_x.append([tmp_x[i],target_x_traj[i]])
            l1_y.append([tmp_y[i],target_y_traj[i]])
            phi1.append([np.sin(b), -np.cos(b)])
            
        if j == 1:
            a2_x.append(tmp_x[i])
            a2_y.append(tmp_y[i])
            l2_x.append([tmp_x[i],target_x_traj[i]])
            l2_y.append([tmp_y[i],target_y_traj[i]])
            phi2.append([np.sin(b), -np.cos(b)])
        if j == 2:
            a3_x.append(tmp_x[i])
            a3_y.append(tmp_y[i])
            l3_x.append([tmp_x[i],target_x_traj[i]])
            l3_y.append([tmp_y[i],target_y_traj[i]])
            phi3.append([np.sin(b), -np.cos(b)])
        if j == 3:
            a4_x.append(tmp_x[i])
            a4_y.append(tmp_y[i])
            l4_x.append([tmp_x[i],target_x_traj[i]])
            l4_y.append([tmp_y[i],target_y_traj[i]])
            phi4.append([np.sin(b), -np.cos(b)])

#Compute the trend of k(Phi)

for i in range(len(phi1)):
    phi[0] = phi1[i]
    phi[1] = phi2[i]
    phi[2] = phi3[i]
    #phi[3] = phi4[i]

    cost = computeCost(phi)
    list_phi.append(cost)

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
math_vars2 = ['\\sigma_m']
ax4.text(10,78,'Simulation Time: 500 (s) \n Accelerated view x100',fontsize=2*tw/3)
ax4.text(10,60,'Packet Delivery Ratio: 90 %',fontsize=2*tw/3)
ax4.text(10,40,'Measurament noise '+r'$ %s $'%math_vars2[0]+' = 5 (deg)',fontsize=2*tw/3)
ax4.text(10,20,'TDMA slot time: 4 (sec)',fontsize=2*tw/3)
ax4.text(10,2,'Average optimization time = '+str(np.round(avgTimeOpt,3))+' (sec)',fontsize=2*tw/3)
#ax4.set_title('Simulation Parameters',y=-0.01)
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        
#ax4.grid()

############################ PLOT TRACKING ERROR #############################
for i in range(int(auvNum)):

    t_prova = np.linspace(0,elapsed_t,len(tracking_errors[i]))
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

ax3.set_ylim([0,max_value])
ax3.set_xlabel('t (s)', labelpad=0.01)
ax3.set_ylabel('RMSE (m)')
ax3.legend(fontsize=fs/3, loc='upper right')
ax3.grid()

#################### PLOT LOSS FUNCTION #######################
tmp = np.linspace(0,elapsed_t,samples)
t = []
for i in range(samples):
    t.append(tmp[i])

y = ax2.plot(t[0],list_phi[0],linewidth=lw,label=r'$ %s $'%math_vars[3])[0]

opt_value = []
for i in range(samples):
    opt_value.append(1)
ax2.plot(t,opt_value,'r--',linewidth=lw,label='Optimal Value')
ax2.set_xlabel('t (s)',labelpad=0.01)
#ax2.set_ylabel(r'$ %s $'%math_vars[3], fontsize=fs)
ax2.legend(fontsize=fs/5,loc='upper right')
ax2.grid()

################## PLOT SIMULATION SCENARIO ###################################
#ax1.set_facecolor('azure')
target = ax1.plot(target_x_traj[0],target_y_traj[0],'r',linewidth=lw,label=r'$ %s $'%math_vars[0]+'(t)')[0]

a1 = ax1.plot(a1_x[0],a1_y[0],'b',label=r'$ %s $'%math_vars[2]+'(t)',linewidth=lw/2)[0]
a2 = ax1.plot(a2_x[0],a2_y[0],'b',linewidth=lw/2)[0]
a3 = ax1.plot(a3_x[0],a3_y[0],'b',linewidth=lw/2)[0]

s1 = ax1.scatter(a3_x[0],a3_y[0],c='b',linewidths=lw)
s2 = ax1.scatter(a3_x[0],a3_y[0],c='b',linewidths=lw)
s3 = ax1.scatter(a3_x[0],a3_y[0],c='b',linewidths=lw)

'''ax1.text(a1_x[0]+20,a1_y[0],r'$ %s $'%sensors_vars[0]+r'$ %s $'%time_vars[0],fontsize=tw)
ax1.text(a2_x[0]+20,a2_y[0],r'$ %s $'%sensors_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)
ax1.text(a3_x[0]+20,a3_y[0],r'$ %s $'%sensors_vars[2]+r'$ %s $'%time_vars[0],fontsize=tw)'''
ax1.text(a1_x[0]+20,a1_y[0],r'$ %s $'%sensors_vars[0],fontsize=tw)
ax1.text(a2_x[0]+20,a2_y[0],r'$ %s $'%sensors_vars[1],fontsize=tw)
ax1.text(a3_x[0]+20,a3_y[0],r'$ %s $'%sensors_vars[2],fontsize=tw)

l1 = ax1.plot(l1_x[0],l1_y[0],'g--',linewidth=lw/4,label='LOS')[0]
l2 = ax1.plot(l2_x[0],l2_y[0],'g--',linewidth=lw/4)[0]
l3 = ax1.plot(l3_x[0],l3_y[0],'g--',linewidth=lw/4)[0]
#l4 = ax1.plot(l4_x[0],l4_y[0],'g--',linewidth=lw/4)[0]

# Static plots - subplot1
ax1.scatter(target_x_traj[0],target_y_traj[0],c='r',label=r'$ %s $'%math_vars[0]+r'$ %s $'%time_vars[0],linewidths=lw)

for i in range(int(auvNum)):
    ax1.scatter(auv_x_traj[0,i],auv_y_traj[0,i],c='b',linewidths=lw)

ax1.scatter(auv_x_traj[0,-1],auv_y_traj[0,-1],c='b',linewidths=lw,label=r'$ %s $'%math_vars[2]+r'$ %s $'%time_vars[0])
estimation = ax1.scatter(x_hat_x[0],x_hat_y[0],c='azure',edgecolors='y',label=r'$ %s $'%math_vars[1]+'(t)',linewidths=lw)

'''cov = Ellipse(xy=(x_hat_x[0],x_hat_y[0]), width=cov_x[0]*2, height=cov_y[0]*2, 
                        edgecolor='r', fc='None', lw=lw)'''
ax1.set_xlabel('x (m)',fontsize=fs/3)
ax1.set_ylabel('y (m)',fontsize=fs/3)
ax1.grid()
ax1.axis('equal')
ax1.legend(fontsize=fs/2)
ax1.set_xlim([-800,+800])
ax1.set_ylim([-800,+800])
'''ax1.set_xlim([-200,+200])
ax1.set_ylim([-200,+200])'''

def update(frame):
    

    # Normalize frame to total samples (for synchronous evolution)
    norm_frame_1 = int(frame / samples * len(tracking_errors[0]))
    norm_frame_2 = int(frame / samples * len(tracking_errors[1]))
    norm_frame_3 = int(frame / samples * len(tracking_errors[2]))
    norm_frame_est = int(frame / 100 * len(x_hat_x))

    y.set_data(t[:frame],list_phi[:frame])
    
    err1.set_data(t_prova[:norm_frame_1],tmp_err1[:norm_frame_1])
    err2.set_data(t_prova[:norm_frame_2],tmp_err2[:norm_frame_2])
    err3.set_data(t_prova[:norm_frame_3],tmp_err3[:norm_frame_3])
    
    # update the line plot:
    target.set_data(target_x_traj[:frame],target_y_traj[:frame])
    a1.set_data(a1_x[:frame],a1_y[:frame])
    a2.set_data(a2_x[:frame],a2_y[:frame]) #if you use : the plot remain
    a3.set_data(a3_x[:frame],a3_y[:frame])
    #a4.set_data(a4_x[:frame],a4_y[:frame])
    data = np.stack([a1_x[frame], a1_y[frame]]).T
    s1.set_offsets(data)
    data = np.stack([a2_x[frame], a2_y[frame]]).T
    s2.set_offsets(data)
    data = np.stack([a3_x[frame], a3_y[frame]]).T
    s3.set_offsets(data)

    l1.set_data(l1_x[frame],l1_y[frame]) # if you dont use : the plot is deleted each iteration
    l3.set_data(l3_x[frame],l3_y[frame])
    if frame >= elapsed_t/2 and AUV_failure == True:
        l2.set_data(0,0)
    else:
        l2.set_data(l2_x[frame],l2_y[frame])
        
      
    data = np.stack([x_hat_x[:norm_frame_est], x_hat_y[:norm_frame_est]]).T
    estimation.set_offsets(data)
    #plt.gca().relim()
    #plt.gca().autoscale_view()

    return (y, target, a1,a2,a3,s1,s2,s3,l1,l2,l3, err1,err2,err3)#a4


plt.yticks(fontsize=fs/3, rotation=0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=fs/3, rotation=0)#to set dimension and orientation of tick labels
ax1.yaxis.set_label_position("right")
ani = animation.FuncAnimation(fig=fig, func=update,
                               frames=samples, interval=50, blit=True)

#ani.save(filename="/home/andrea/animations/fixed_target.mp4", writer='ffmpeg', fps=30,dpi=200)  # Increase DPI for better quality)
            
#ani2.save(filename="/home/andrea/animations/fixed_target_cost.gif", writer="pillow")

plt.show()
