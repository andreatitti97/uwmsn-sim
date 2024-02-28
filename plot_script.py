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

for i in range(int(auvNum)):
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_x_traj.txt')
    auv_y_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_y_traj.txt')

    tmp = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_1.txt')
    x_hat_ = np.zeros((len(tmp),4))
    err = np.loadtxt(log_directory+'/'+str(i+1)+'-err.txt')
    cov = np.zeros((len(tmp),4))

    
    for j in range(4):
        x_hat_[:,i] = np.loadtxt(log_directory+'/'+str(i+1)+'-x_hat_'+str(j+1)+'.txt')
        cov[:,i] = np.loadtxt(log_directory+'/'+str(i+1)+'-cov'+str(j+1)+'.txt')


    x_hat.append(x_hat_)
    P.append(cov)
    tracking_errors.append(err)

# PLOTs
lw_ms = 2*7
#fig = plt.figure(1)
#patch = fig.patch 
fig, ax = plt.subplots()

ax.set_facecolor('cornflowerblue')
plt.title('SCENARIO')

for i in range(int(auvNum)):
    plt.plot(auv_x_traj[:,i],auv_y_traj[:,i],'b',markersize=lw_ms)
    
plt.plot(target_x_traj,target_y_traj)

#plt.plot.set_facecolor('cornflowerblue')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
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
    plt.grid()
plt.show()

    

#ADD automated image saving