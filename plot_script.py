#!+usr+bin+env python
import matplotlib.pyplot as plt
import numpy as np
import os, importlib
import pathlib
from math import pi
from matplotlib.lines import Line2D

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

auv_x_traj = np.zeros((len(target_x_traj),int(auvNum)))
auv_y_traj = np.zeros((len(target_y_traj),int(auvNum)))

x_hat = []
tracking_errors= []
P = []
PDR = np.zeros((int(auvNum),1))

for i in range(int(auvNum)):
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_x_traj.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_y_traj.txt')

    


    tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_2.txt')
    x_hat_ = np.zeros((len(tmp),4))
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-err.txt')
    cov = np.zeros((len(tmp),4))
 
    PDR[i] = np.loadtxt(log_directory+'/'+str(i+1)+'-PDR')
    
    for j in range(4):
        x_hat_[:,i] = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_'+str(j+1)+'.txt')
        cov[:,i] = np.loadtxt(log_directory+'/'+str(i+1)+'-cov'+str(j+1)+'.txt')



    x_hat.append(x_hat_)
    P.append(cov)
    tracking_errors.append(err)

# PLOTs
print('SIMULATION INFO [auvNum - Simulation Time (s) - Slot Time (s)]',sim_info)   
print('Acoustic Communication Stat [PDR AUV1,PDR AUV2,PDR AUV3,PDR AUV4]:',PDR)
for i in range(int(auvNum)):
    print('AUV ID %s RMSE (m):%s',i+1,(sum(tracking_errors[i])/len(tracking_errors[i])))
lw_ms = 2*7
#fig = plt.figure(1)
#patch = fig.patch 
fig, ax = plt.subplots()

ax.set_facecolor('cornflowerblue')
plt.title('SCENARIO')


tot_smpls = len(target_x_traj)
scaler = 5000
plt.plot(target_x_traj,target_y_traj)
plt.plot(target_x_traj[0],target_y_traj[0],'ro',markersize=lw_ms)
plt.plot(target_x_traj[-1],target_y_traj[-1],'ro',markersize=lw_ms)
for i in range(int(auvNum)):
    plt.plot(auv_x_traj[-1,i],auv_y_traj[-1,i],'og',markersize=lw_ms)
    plt.plot([auv_x_traj[-1,i],
            target_x_traj[-1]],[auv_y_traj[-1,i],target_y_traj[-1]],'g--',linewidth=2)
    plt.plot(auv_x_traj[0,i],auv_y_traj[0,i],'ob',markersize=lw_ms)
    
    plt.plot(auv_x_traj[:,i],auv_y_traj[:,i],'b',markersize=lw_ms)
    
    '''for j in range(int(tot_smpls/scaler)):
        # plot LOS
        idx = (j+1)*scaler
        
        plt.plot([auv_x_traj[idx,i],
            target_x_traj[idx]],[auv_y_traj[idx,i],target_y_traj[idx]],'k--',linewidth=1)'''
    


#plt.plot.set_facecolor('cornflowerblue')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid()
plt.axis('equal')
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
plt.show()


    

#ADD automated image saving