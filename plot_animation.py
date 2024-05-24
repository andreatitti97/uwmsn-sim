#!+usr+bin+env python
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
import os, pathlib
import matplotlib.animation as animation

# Environment initialization
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/uwmsn-sim'
log_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/logs'
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
    PDR[i] = np.loadtxt(log_directory+'/'+str(i+1)+'-PDR')

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
for i in range(int(auvNum)):
    print('OPTIMIZATION STATS --> Average Optimization Time AUV ID:',i+1,sum(avgTime[i])/len(avgTime[i]))
    print('AUV ID RMSE (m):',i+1,(sum(tracking_errors[i])/len(tracking_errors[i])))

###############################################
# Animated plot

# Parameters
lw = 8
fs = 50

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
time_vars = ['(t_{0})','(t_{f})']

# Conditioning - Figure 2
fig2, ax2 = plt.subplots(1,1)
plt.yticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels

tmp = np.linspace(0,elapsed_t,samples)
t = []
for i in range(samples):
    t.append(tmp[i])

y = ax2.plot(t[0],list_phi[0],linewidth=lw,label=r'$ %s $'%math_vars[3])[0]

opt_value = []
for i in range(samples):
    opt_value.append(1)
ax2.plot(t,opt_value,'r--',linewidth=lw,label='Optimal Value')

def update_cost(frame):
    y.set_data(t[:frame],list_phi[:frame])

    return(y,)

print(len(t))

ax2.set_xlabel('t (s)', fontsize = fs)
ax2.set_ylabel(r'$ %s $'%math_vars[3], fontsize=fs)
ax2.legend(fontsize=fs/2)
ax2.grid()

ani2 = animation.FuncAnimation(fig=fig2, func=update_cost,
                               frames=samples, interval=5, blit=False)

# simulation Scenario - Figure 1

fig, ax = plt.subplots(1,1)
plt.yticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
ax.set_facecolor('azure')
target = ax.plot(target_x_traj[0],target_y_traj[0],'r',linewidth=lw,label=r'$ %s $'%math_vars[0]+'(t)')[0]

a1 = ax.plot(a1_x[0],a1_y[0],'b',label=r'$ %s $'%math_vars[2]+'(t)',linewidth=lw)[0]
a2 = ax.plot(a2_x[0],a2_y[0],'b',linewidth=lw)[0]
a3 = ax.plot(a3_x[0],a3_y[0],'b',linewidth=lw)[0]
#a4 = ax.plot(a4_x[0],a4_y[0],'b',linewidth=lw)[0]

l1 = ax.plot(l1_x[0],l1_y[0],'g--',linewidth=lw/4,label='LOS')[0]
l2 = ax.plot(l2_x[0],l2_y[0],'g--',linewidth=lw/4)[0]
l3 = ax.plot(l3_x[0],l3_y[0],'g--',linewidth=lw/4)[0]
#l4 = ax.plot(l4_x[0],l4_y[0],'g--',linewidth=lw/4)[0]

# Static plots - subplot1
ax.scatter(target_x_traj[0],target_y_traj[0],c='r',label=r'$ %s $'%math_vars[0]+r'$ %s $'%time_vars[0],linewidths=lw)

for i in range(int(auvNum)):
    ax.scatter(auv_x_traj[0,i],auv_y_traj[0,i],c='b',linewidths=lw)

ax.scatter(auv_x_traj[0,-1],auv_y_traj[0,-1],c='b',linewidths=lw,label=r'$ %s $'%math_vars[2]+r'$ %s $'%time_vars[0])
estimation = ax.scatter(x_hat_x[0],x_hat_y[0],c='azure',edgecolors='y',label=r'$ %s $'%math_vars[1]+'(t)',linewidths=lw)

def update(frame):
    
    # update the line plot:
    target.set_data(target_x_traj[:frame],target_y_traj[:frame])
    a1.set_data(a1_x[:frame],a1_y[:frame])
    a2.set_data(a2_x[:frame],a2_y[:frame]) #if you use : the plot remain
    a3.set_data(a3_x[:frame],a3_y[:frame])
    #a4.set_data(a4_x[:frame],a4_y[:frame])
    
    l1.set_data(l1_x[frame],l1_y[frame]) # if you dont use : the plot is deleted each iteration
    l2.set_data(l2_x[frame],l2_y[frame])
    l3.set_data(l3_x[frame],l3_y[frame])
    #l4.set_data(l4_x[frame],l4_y[frame])


    if frame % est_samples == 0:  

        data = np.stack([x_hat_x[:int(frame/est_samples)], x_hat_y[:int(frame/est_samples)]]).T
        estimation.set_offsets(data)
        
    #plt.gca().relim()
    #plt.gca().autoscale_view()

    return (target, a1,a2,a3,l1,l2,l3,estimation)#a4

print(len(target_x_traj))
# you can animate multiple fiures simultaneosuly
ani = animation.FuncAnimation(fig=fig, func=update,
                               frames=samples, interval=5, blit=False)

ax.set_xlabel('x (m)',fontsize=fs)
ax.set_ylabel('y (m)',fontsize=fs)
ax.grid()
ax.axis('equal')
ax.legend(fontsize=fs/2)
ax.set_xlim([-800,+800])
ax.set_ylim([-800,+800])

plt.show()






''' UTILS
plt.xlim([-400,250])
plt.ylim([-250,400])
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
ani.save(filename="/home/andrea/animations/test1.gif", writer="pillow")
'''