#!+usr+bin+env python
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
import os, pathlib
from scipy.interpolate import make_interp_spline

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


# Downsampling script
original_samples = len(target_x_traj)
sampling = 200#100 ideal #200 realistic scneario
tmp_x = target_x_traj
target_x_traj = tmp_x[::sampling]
tmp_y = target_y_traj
target_y_traj = tmp_y[::sampling]
samples = len(target_x_traj)

auv_x = np.zeros((samples,int(auvNum)))
auv_y = np.zeros((samples,int(auvNum)))
x = np.linspace(0,elapsed_t,original_samples) 
t = np.linspace(0,elapsed_t,samples)#subsampled set
for i in range(int(auvNum)):
    tmp_x = auv_x_traj[:,i]
    model = make_interp_spline(x, auv_x_traj[:,i],k=9) #TODO CHECK ROUNDING UP PROBLEM FOR INTERP
    auv_x[:,i] = model(t)

    tmp_y = auv_y_traj[:,i]

    model = make_interp_spline(x, auv_y_traj[:,i],k=9) #TODO CHECK ROUNDING UP PROBLEM FOR INTERP
    auv_y[:,i] = model(t)

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

# Compute the trend of k(Phi)
phi = np.zeros((4,2))
list_phi, phi1, phi2, phi3, phi4 = [], [], [], [], []

for j in range(int(auvNum)):
    tmp_x = auv_x[:,j]
    tmp_y = auv_y[:,j]
    for i in range(samples):
        b = atan2(tmp_y[i]-target_y_traj[i],tmp_x[i]-target_x_traj[i])
        if j == 0:
            phi1.append([np.sin(b), -np.cos(b)])
        if j == 1:
            phi2.append([np.sin(b), -np.cos(b)])
        if j == 2:
            phi3.append([np.sin(b), -np.cos(b)])
        if j == 3:
            phi4.append([np.sin(b), -np.cos(b)])

for i in range(len(phi1)):
    phi[0] = phi1[i]
    phi[1] = phi2[i]
    phi[2] = phi3[i]
    #phi[3] = phi4[i]

    cost = computeCost(phi)
    list_phi.append(cost)
#np.savetxt('/home/andrea/Documents/controlo_paper_results/official_results/validation1/logs/logs_COMPARISON_ERRORS/cond_range3',list_phi)
#np.savetxt('/home/andrea/Documents/controlo_paper_results/official_results/validation1/logs/logs_COMPARISON_ERRORS/err_range3',tracking_errors[0])

##########################################################
# PLOT SETUP
fs = 42
lw = 5
sw = 1
tw = 40
math_vars = ['\\kappa(\\Phi)','\\xi']
agents_vars = ['s_1','s_2','s_3','s_4']
time_vars = ['(t_{0})','(t_{f})','(t_{0}=t_{f})']


##########################################################
# Plot trend conditioning estimation problem
fig, ax = plt.subplots()
t_axis = np.linspace(0,elapsed_t,len(list_phi))
max = max(list_phi)
tmp = []
for i in range(samples):
    tmp.append(list_phi[i]/max)

ax.plot(t_axis,list_phi,label=r'$ %s $'%math_vars[0],linewidth=lw)
opt_value = []
for i in range(samples):
    opt_value.append(1)
ax.plot(t_axis,opt_value,'r--',label='Optimal Value',linewidth=lw)
ax.set_xlabel('t (s)', fontsize = fs)
ax.set_ylabel(r'$ %s $'%math_vars[0], fontsize=fs)
ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels
#plt.show()

##########################################################
# Plot simulation scenario

fig2, ax = plt.subplots()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels
test = []
a = 0
for i in range(samples):
    a = 0 + i#samples - i
    test.append(a*10)

c_map = ax.scatter(target_x_traj,target_y_traj,c=test,cmap='autumn_r',vmin=0, vmax=elapsed_t,linewidths=sw)
ax.scatter(target_x_traj[0],target_y_traj[0],c='y',marker='o',linewidths=sw*8)
ax.scatter(target_x_traj[-1],target_y_traj[-1],c='k',linewidths=sw*12)
ax.scatter(target_x_traj[-1],target_y_traj[-1],c='r',linewidths=sw*8)

#ax.text(target_x_traj[-1],target_y_traj[-1]-23,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)
ax.text(target_x_traj[-1]+6,target_y_traj[-1]-18,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
ax.text(target_x_traj[0]+15,target_y_traj[0],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)


cb = fig.colorbar(c_map, ax=ax)
cb.set_label('t (s)',fontsize=fs)
cb.ax.tick_params(labelsize=(fs/3)*2)

for i in range(int(auvNum)):
    
    if i == 0:
        ax.plot([auv_x[-1,i],
            target_x_traj[-1]],[auv_y[-1,i],target_y_traj[-1]],'r--',linewidth=lw/3,label='LOS'+r'$ %s $'%time_vars[1])
    else:
        ax.plot([auv_x[-1,i],
            target_x_traj[-1]],[auv_y[-1,i],target_y_traj[-1]],'r--',linewidth=lw/3)
        
    if i == 0:
        a,b,c,d = +5,+0,+10,0
    elif i == 1:
        a,b,c,d = -45,+13,-0,-35
    else:
        a,b,c,d = +15,-8,-30,+30

    ax.scatter(auv_x[:,i],auv_y[:,i],c=test,cmap='autumn_r',linewidths=sw)
    ax.text(auv_x[0,i]+a,auv_y[0,i]+b,r'$ %s $'%agents_vars[i]+r'$ %s $'%time_vars[0],fontsize=tw)
    ax.scatter(auv_x[0,i],auv_y[0,i],c='y',linewidths=sw*8)
    #ax.text(auv_x[-1,i]+c,auv_y[-1,i]+d,r'$ %s $'%agents_vars[i]+r'$ %s $'%time_vars[1],fontsize=tw)
    ax.scatter(auv_x[-1,i],auv_y[-1,i],c='r',linewidths=sw*8)


'''PROPOSAL
ax.text(target_x_traj[-1],target_y_traj[-1]+5,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)
if i == 0:
        a,b,c,d = +5,+0,+2,0
    elif i == 1:
        a,b,c,d = +5,+0,-0,+5
    else:
        a,b,c,d = -5,-8,-3,+5'''

'''RANGE

ax.text(target_x_traj[-1]-5,target_y_traj[-1]+4,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[2],fontsize=(tw/3)*2)
if i == 0:
        a,b,c,d = +5,-10,-10,-10
    elif i == 1:
        a,b,c,d = +5,+0,-3,+8
    else:
        a,b,c,d = -10,-8,0,-7'''

'''GEOM
ax.text(target_x_traj[-1]+6,target_y_traj[-1]-18,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
ax.text(target_x_traj[0]+15,target_y_traj[0],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)
if i == 0:
        a,b,c,d = +5,0,0,+12
    elif i == 1:
        a,b,c,d = 0,+10,-50,-5
    else:
        a,b,c,d = +15,0,0,+5'''


'''REALISTIC
ax.text(target_x_traj[-1]+6,target_y_traj[-1]-18,r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[1],fontsize=tw)
ax.text(target_x_traj[0]+15,target_y_traj[0],r'$ %s $'%math_vars[1]+r'$ %s $'%time_vars[0],fontsize=tw)


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
ax.legend(fontsize=(fs*2)/3,loc='lower right')
#plt.show()

##########################################################
# Plot tracking error
fig, ax = plt.subplots()
math_vars = ['AUV1','AUV2','AUV3','\\epsilon']
for i in range(int(auvNum)):

    t = np.linspace(0,500,len(tracking_errors[i]))
    tmp = tracking_errors[i]
    ax.plot(t,tmp,label=r'$ %s $'%math_vars[i],linewidth=lw)
    max_value = tmp.max()

epsi = []    
for i in range(len(t)):
    epsi.append(0.0)
ax.plot(t,epsi,'r--',linewidth=lw,label=r'$ %s $'%math_vars[-1])    
ax.set_xlabel('t (s)', fontsize=fs)
ax.set_ylabel('RMSE (m)',fontsize=fs)
ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels
#plt.show()


##########################################################
# Plot relative distance between auvs and target

fig5, ax = plt.subplots()
sampling = 1
math_vars = ['d_1','d_2','d_3','d_4','d_{-}^{r}','d_{+}^{r}']
lthres = 50
hthresh = 150

x = np.linspace(0,elapsed_t,int(samples/sampling)) #subsampled set

for i in range(int(auvNum)):
    #plt.subplot(int(auvNum),1,i+1)
    dist = []
    low_thresh = []
    high_thresh = []
    
    
    for j in range(samples):

        low_thresh.append(lthres)
        high_thresh.append(hthresh)
        if i == 0 or i == 2:
            a = 25
        else:
            a = 0
        dist.append(np.sqrt((target_x_traj[j]-tmp_x[j])**2+(target_y_traj[j]-tmp_y[j])**2)-a)

    
    model = make_interp_spline(x, dist[::sampling])
    t = np.linspace(0,elapsed_t,samples)#original samples length but interpolated
    y = model(t)
    
    ax.plot(t,y,linewidth=lw,label=r'$ %s $'%math_vars[i])

ax.set_xlabel('t (s)',fontsize=fs)
ax.set_ylabel('d (m)',fontsize=fs)
ax.plot(t,low_thresh,'r--',linewidth=lw/2,label=r'$ %s $'%math_vars[4])
ax.plot(t,high_thresh,'g--',linewidth=lw/2,label=r'$ %s $'%math_vars[5])
ax.legend(fontsize=fs*2/3)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels


##########################################################
# Plot Surge Velocities

sampling = 1
fig3, ax = plt.subplots()

for i in range(int(auvNum)):

    tmp = surge_vel[i]

    x = np.linspace(0,elapsed_t,int(len(tmp)/sampling)) #subsampled set
    model = make_interp_spline(x, tmp[::sampling])
    t = np.linspace(0,elapsed_t,samples)#original samples length but interpolated
    y = model(t)

    ax.plot(t,y,label='AUV'+str(i+1),linewidth=lw)
    
ax.set_ylabel('u (m/s)', fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels

##########################################################
# Plot distances between the AUVs
#fig6, (ax1, ax2, ax3) = plt.subplots(int(auvNum),1)
fig6, ax = plt.subplots()
sampling = 1
math_vars = ['d_{ij}']
for i in range(int(auvNum)):
    
    dist_ij1= []
    dist_ij2 = []
    
    if i == 0:
        idx1 = 1
        idx2 = 2
    elif i == 1:
        idx1 = 1
        idx2 = 2
    else:
        idx1 = 1
        idx2 = 2
    tmp_x_i = auv_x[:,i]
    tmp_y_i = auv_y[:,i]
    tmp_x_j1 = auv_x[:,idx1]
    tmp_y_j1 = auv_y[:,idx1]
    tmp_x_j2 = auv_x[:,idx2]
    tmp_y_j2 = auv_y[:,idx2]
    loops = len(tmp_x)
    for j in range(loops):
        dist_ij1.append(np.sqrt((tmp_y_i[j]-tmp_x_j1[j])**2+(tmp_y_i[j]-tmp_y_j1[j])**2))
        dist_ij2.append(np.sqrt((tmp_y_i[j]-tmp_x_j2[j])**2+(tmp_y_i[j]-tmp_y_j2[j])**2))
    
    ax.plot(t,dist_ij1,label='dist AUV'+str(i+1)+'--'+str(idx1+1))
    ax.plot(t,dist_ij2,label='dist AUV'+str(i+1)+'--'+str(idx2+1))

ax.set_ylabel(r'$ %s $'%math_vars[0], fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs)/3, rotation=0)#to set dimension and orientation of tick labels
plt.show()
##########################################################
# Plot heading 
fig4, ax = plt.subplots()
theta_var = ['\\theta_{s}']
sampling = 1
for i in range(int(auvNum)):
    tmp = heading[i]
    for j in range(len(tmp)):
        if tmp[j] < 0:
            tmp[j] = 360 + tmp[j]      

    x = np.linspace(0,elapsed_t,int(len(tmp)/sampling)) #subsampled set

    model = make_interp_spline(x, tmp[::sampling]) #TODO CHECK ROUNDING UP PROBLEM FOR INTERP
    t = np.linspace(0,elapsed_t,samples)#original samples length but interpolated
    y = model(t)

    ax.plot(t,y,label='AUV'+str(i+1),linewidth=lw)

ax.set_ylabel(r'$ %s $'%theta_var[0]+'(rad)', fontsize=fs)
ax.set_xlabel('t (s)', fontsize =fs)
ax.legend(fontsize=fs)
ax.grid()
plt.yticks(fontsize=(fs*2)/3, rotation = 0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=(fs*2)/3, rotation=0)#to set dimension and orientation of tick labels
#plt.show()

'''Interpolation script
x = np.linspace(0,elapsed_t,int(len(target_x_traj)/500)+1) #subsampled set
    
    model = make_interp_spline(x, dist[::500])

    t = np.linspace(0,elapsed_t,len(target_x_traj))#original samples length but interpolated
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