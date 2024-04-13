#!+usr+bin+env python
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
import os, pathlib
from matplotlib.lines import Line2D #can b used for costum legends
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from scipy import interpolate

pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/uwmsn-sim'
log_directory = pkg_directory+'/logs'
class_directory = pkg_directory+'/src'+'/Classes'

sim_info = np.loadtxt(log_directory+'/sim_info.txt')
ctrl_set = np.loadtxt(log_directory+'/ctrl_set.txt')
auvNum = sim_info[0]
elapsed_t = sim_info[1]

Ts = sim_info[2]

target_x_traj = np.loadtxt(log_directory+'/target_x_traj.txt')
target_y_traj = np.loadtxt(log_directory+'/target_y_traj.txt')

target_x_traj2 = np.loadtxt(log_directory+'/target_x_traj.txt')
target_y_traj2 = np.loadtxt(log_directory+'/target_y_traj.txt')



auv_x_traj = np.zeros((len(target_x_traj),int(auvNum)))
auv_y_traj = np.zeros((len(target_y_traj),int(auvNum)))
surge_vel = [[],[],[],[]]
heading = [[],[],[],[]]

x_hat, tracking_errors, P, avgNodes, avgTime = [], [], [], [], []

PDR = np.zeros((int(auvNum),1))


for i in range(int(auvNum)):
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_x_traj.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_y_traj.txt')

    avgTime.append(np.loadtxt(log_directory+'/wall_times'+str(i+1)+'.txt')) 
    avgNodes.append(np.loadtxt(log_directory+'/nodes'+str(i+1)+'.txt'))    

    tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_2.txt')
    x_hat_ = np.zeros((len(tmp),4))
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-err.txt')
    cov = np.zeros((len(tmp),4))
    
    surge_vel[i] = np.loadtxt(log_directory+'/'+str(i+1)+'surge_vel')
    heading[i] = np.loadtxt(log_directory+'/'+str(i+1)+'heading')*180/np.pi


    PDR[i] = np.loadtxt(log_directory+'/'+str(i+1)+'-PDR')
    
    for j in range(4):
        x_hat_[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_'+str(j+1)+'.txt')
        cov[:,j] = np.loadtxt(log_directory+'/'+str(i+1)+'-cov'+str(j+1)+'.txt')
     

    x_hat.append(x_hat_)
    P.append(cov)
    tracking_errors.append(err)



cov_x = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(1)+'.txt')
cov_y = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(2)+'.txt')   
cov_vx = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(3)+'.txt') 
cov_vy = np.loadtxt(log_directory+'/'+str(1)+'-cov'+str(4)+'.txt') 
x_hat_x = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(1)+'.txt')
x_hat_y = np.loadtxt(log_directory+'/'+str(1)+'-x_hat_'+str(2)+'.txt')

def computeCost(phi):

    length_y = len(phi)

    
    W = np.zeros((length_y,length_y))
    for i in range(length_y):
        W[i,i] = 1.0

    PHI = np.dot(np.transpose(phi),np.dot(np.linalg.inv(W),phi))

    return np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)
    

# PLOTs

'''def sig(x):
    
    alpha = -0.1
    gamma = 80
    return 1/(1+np.e**(alpha*(gamma-x)))

x = np.linspace(0,100)
plt.plot(x,sig(x))
plt.grid()
plt.show()'''

print('SIMULATION INFO [auvNum - Simulation Time (s) - Slot Time (s)]',sim_info)   
print('Acoustic Communication Stat [PDR AUV1,PDR AUV2,PDR AUV3,PDR AUV4]:',PDR)
for i in range(int(auvNum)):
    print('OPTIMIZATION STATS --> Average Optimization Time AUV ID:',i+1,sum(avgTime[i])/len(avgTime[i]))
    print('AUV ID RMSE (m):',i+1,(sum(tracking_errors[i])/len(tracking_errors[i])))

fig, ax = plt.subplots(1,1)
ax.set_facecolor('lightskyblue')

tot_smpls = len(target_x_traj)
scaler = 5000
lw_ms = 2*7


tmp_x = target_x_traj[::10]
target_x_traj = tmp_x
tmp_y = target_y_traj[::10]
target_y_traj = tmp_y


a1_x, a2_x, a3_x, a4_x = [], [], [], []
a1_y, a2_y, a3_y, a4_y = [], [], [], []

l1_x, l2_x, l3_x, l4_x = [], [], [], []
l1_y, l2_y, l3_y, l4_y = [], [], [], []

phi = np.zeros((4,2))
list_phi, phi1, phi2, phi3, phi4 = [], [], [], [], []

for j in range(int(auvNum)):
    tmp_x = auv_x_traj[:,j]
    tmp_x = tmp_x[::10]
    tmp_y = auv_y_traj[:,j]
    tmp_y = tmp_y[::10]
    
    
    for i in range(len(tmp_x)):
        b = atan2(tmp_y[i]-target_y_traj[i],tmp_x[i]-target_x_traj[i])
        
        #phi[0,:] = [np.sin(b), -np.cos(b)]
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

for i in range(len(phi1)):
    phi[0] = phi1[i]
    phi[1] = phi2[i]
    phi[2] = phi3[i]
    phi[3] = phi4[i]
    cost = computeCost(phi)
    list_phi.append(cost)

# Moving Plots - subplot1
target = ax.plot(target_x_traj[0],target_y_traj[0],'r',markersize=lw_ms,label='x(t)')[0]

a1 = ax.plot(a1_x[0],a1_y[0],'b',label='s'+str(1)+'(t)')[0]
a2 = ax.plot(a2_x[0],a2_y[0],'b',label='s'+str(2)+'(t)')[0]
a3 = ax.plot(a3_x[0],a3_y[0],'b',label='s'+str(3)+'(t)')[0]
a4 = ax.plot(a4_x[0],a4_y[0],'b',label='s'+str(4)+'(t)')[0]

l1 = ax.plot(l1_x[0],l1_y[0],'g--',label='LOS')[0]
l2 = ax.plot(l2_x[0],l2_y[0],'g--')[0]
l3 = ax.plot(l3_x[0],l3_y[0],'g--')[0]
l4 = ax.plot(l4_x[0],l4_y[0],'g--')[0]

# Static plots - subplot1
ax.plot(target_x_traj[0],target_y_traj[0],'ro',markersize=lw_ms,label='s(t0)')

for i in range(int(auvNum)):
    ax.plot(auv_x_traj[0,i],auv_y_traj[0,i],'ob',markersize=lw_ms,label='s'+str(i+1)+'(t0)')

estimation = ax.scatter(x_hat_x[0],x_hat_y[0],c='lightskyblue',edgecolors='r',label='x_hat(t)')

# Moving plots - suplot2
'''
fig2, ax2 = plt.subplots(1,1)
ax2.set_facecolor('lightskyblue')
t_axis = np.linspace(0,elapsed_t,len(list_phi))
obj_func = ax2.plot(t_axis[0],list_phi[0],label='Objective Function')[0]
ax2.set(xlim=[0,elapsed_t],ylim=[0,1],xlabel='t [s]', ylabel='y [m]')
def update2(frame):

    obj_func.set_data(t_axis[:frame],list_phi[:frame])

    return (obj_func)

    
'''
def update(frame):
    
    # update the line plot:

    target.set_data(target_x_traj[:frame],target_y_traj[:frame])
    a1.set_data(a1_x[:frame],a1_y[:frame])
    a2.set_data(a2_x[:frame],a2_y[:frame])
    a3.set_data(a3_x[:frame],a3_y[:frame])
    a4.set_data(a4_x[:frame],a4_y[:frame])
    
    l1.set_data(l1_x[frame],l1_y[frame])
    l2.set_data(l2_x[frame],l2_y[frame])
    l3.set_data(l3_x[frame],l3_y[frame])
    l4.set_data(l4_x[frame],l4_y[frame])

    
    if frame % 100 == 0:  

        data = np.stack([x_hat_x[:int(frame/100)], x_hat_y[:int(frame/100)]]).T
        estimation.set_offsets(data)
        
    #plt.gca().relim()
    #plt.gca().autoscale_view()


    return (target, a1,a2,a3,a4,l1,l2,l3,l4,estimation)

ax.set(xlim=[-250,250],ylim=[-250,250],xlabel='x [m]', ylabel='y [m]')

ax.legend()
ax.grid()
ax.axis('equal')

# you can animate multiple fiures simultaneosuly
'''
ani2 = animation.FuncAnimation(fig=fig2, func=update2,
                               frames=len(target_x_traj), interval=1, blit=True)'''
ani = animation.FuncAnimation(fig=fig, func=update,
                               frames=len(target_x_traj), interval=1, blit=True)

plt.xlim([-250,250])
plt.ylim([-250,250])
#manager = plt.get_current_fig_manager()
#manager.full_screen_toggle()
plt.show()
#ani.save(filename="/home/andrea/animations/test1.gif", writer="pillow")
##########################################################
fs = 40
lw = 4
math_vars = ['\\sigma_m','\\xi','\\bar{d}_r']
#PLOT OBJECTIV FUNCTION

fig, ax = plt.subplots(1,1)
t_axis = np.linspace(0,elapsed_t,len(list_phi))
max = max(list_phi)
tmp = []
for i in range(len(list_phi)):
    tmp.append(list_phi[i]/max)

ax.plot(t_axis,list_phi,label='Objective Function',linewidth=lw)
ax.grid()

ax.set_xlabel('t(s)', fontsize = fs)
ax.set_ylabel(r'$ %s $'%math_vars[0], fontsize=fs)
plt.yticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.show()


fig, ax = plt.subplots()

ax.set_facecolor('lightskyblue')
plt.title('Simulation Scenario')

tot_smpls = len(target_x_traj)
scaler = 5000
plt.plot(target_x_traj,target_y_traj)
plt.plot(target_x_traj[0],target_y_traj[0],'ro',markersize=lw_ms)
plt.plot(target_x_traj[-1],target_y_traj[-1],'ro',markersize=lw_ms)
for i in range(int(auvNum)):
    plt.plot(auv_x_traj[-1,i],auv_y_traj[-1,i],'og',markersize=lw_ms)
    plt.plot([auv_x_traj[-1,i],
            target_x_traj[-1]],[auv_y_traj[-1,i],target_y_traj[-1]],'g--',linewidth=lw)
    plt.plot(auv_x_traj[0,i],auv_y_traj[0,i],'ob',markersize=lw_ms)
    
    plt.plot(auv_x_traj[:,i],auv_y_traj[:,i],'b',markersize=lw_ms)
    

    '''for j in range(int(tot_smpls/scaler)):
        # plot LOS
        idx = (j+1)*scaler
        
        plt.plot([auv_x_traj[idx,i],
            target_x_traj[idx]],[auv_y_traj[idx,i],target_y_traj[idx]],'k--',linewidth=1)'''
    


#plt.plot.set_facecolor('cornflowerblue')
plt.xlabel('x (m)',fontsize=fs)
plt.ylabel('y (m)',fontsize=fs)
plt.grid()
plt.axis('equal')

plt.yticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.show()

##########################################################
plt.title('Speed and Heading profiles')

from scipy.interpolate import make_interp_spline
plt.title('Surge Velocity profiles')
# PLot surge vel during the mission
for i in range(int(auvNum)):
    #plt.subplot(int(auvNum),1,i+1)
    
    t = np.linspace(0,elapsed_t,len(surge_vel[i]))
    plt.plot(t,surge_vel[i],label='u'+str(i+1),linewidth=lw)
    
plt.ylabel('u', fontsize=fs)


plt.xlabel('t(s)', fontsize = fs)

plt.yticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.xticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels

plt.grid()
plt.show()

'''# PLOT HEADING DURING MISSION
for i in range(int(auvNum)):
    plt.subplot(int(auvNum),1,i+1)
    tmp = heading[i]
    for j in range(len(tmp)):
        
        if tmp[j] < 0:
            
            tmp[j] = 360 + tmp[j]      
    
    tmp = tmp[::1000]
    x = np.linspace(0,elapsed_t,int(len(tmp)))
    model = make_interp_spline(x, tmp)
    
    t = np.linspace(0,elapsed_t,len(surge_vel[i]))
    y = model(t)
    plt.plot(t,y)
    plt.grid()
plt.show()'''
# Plot relative distance between auvs and target

for i in range(int(auvNum)):
    #plt.subplot(int(auvNum),1,i+1)
    dist = []
    thresh = []
    tmp_x = auv_x_traj[:,i]
    tmp_y = auv_y_traj[:,i]
    #tmp_x = tmp_x[::10]        
    #tmp_y = tmp_y[::10]
    for j in range(len(target_x_traj2)):

        thresh.append(20)
        dist.append(np.sqrt((target_x_traj2[j]-tmp_x[j])**2+(target_y_traj2[j]-tmp_y[j])**2))

    t = np.linspace(0,elapsed_t,len(target_x_traj2))
    plt.ylabel('d'+str(i+1),fontsize=fs)
    plt.yticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
    #plt.ylim([0,90])
    
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

    
    
    plt.plot(t,dist,linewidth=lw,label='d'+str(i+1))
plt.plot(t,thresh,'r--',linewidth=lw/2,label=r'$ %s $'%math_vars[2])
plt.legend()
plt.grid()
#plt.axis('equal')
plt.xlabel('t (s)',fontsize=fs)
plt.xlim([0,450])
plt.ylim([0,200])
plt.xticks(fontsize=fs/2, rotation=0)#to set dimension and orientation of tick labels
plt.show()

'''plt.title('Consensus')
t = np.linspace(0,elapsed_t,len(target_x_traj2))
tmp = x_hat[0]
length = len(tmp[:,0])
for i in range(int(auvNum)):
    #plt.subplot(int(auvNum),1,i+1)
    tmp = x_hat[i]
    if len(tmp[:,0]) >= length:
        length = length
    else:
        length = len(tmp[:,0])
    
t = np.linspace(0,elapsed_t,length)
for i in range(4):
    plt.subplot(int(auvNum),1,i+1)
    for j in range(int(auvNum)): 
        tmp = x_hat[j]
        plt.plot(t,tmp[0:length,i])
    plt.grid()
    
plt.show()

plt.title('Tracking Errors')
for i in range(int(auvNum)):
    plt.subplot(int(auvNum),1,i+1)
    t = np.linspace(0,elapsed_t,len(tracking_errors[i]))
    tmp = tracking_errors[i]
    plt.plot(t,tmp)
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Tracking error')
    max_value = tmp.max()
    plt.text(len(t)/2,max_value-max_value/8,'RMSE (m):'+str((sum(tracking_errors[i])/len(tracking_errors[i]))))
    plt.grid()
plt.show()'''

#ADD automated image saving
