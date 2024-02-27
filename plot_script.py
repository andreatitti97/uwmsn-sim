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
auv_x_traj = np.zeros((len(target_x_traj),int(auvNum)))

for i in range(int(auvNum)):
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_x_traj.txt')
    auv_x_traj[:,i] = np.loadtxt(log_directory+'/auv'+str(i+1)+'_y_traj.txt')

print(auv_x_traj)