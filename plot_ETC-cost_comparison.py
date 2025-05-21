#!+usr+bin+env python
import matplotlib.pyplot as plt
import os, importlib.util, pathlib
import numpy as np
from math import atan2
import os, pathlib
from scipy.interpolate import make_interp_spline

# Environment initialization
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/uwmsn-sim'
log_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())+'/logs_paper-ETC'
class_directory = pkg_directory+'/src'+'/Classes'

# Import config file 
module_dir = os.path.dirname(pathlib.Path(__file__).parent.resolve())
header_file = pkg_directory+'/include'+'/uwmsn-sim'
spec = importlib.util.spec_from_file_location("module.header", header_file+'/main-kinematic_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

desAlpha = [0.001,0.05,0.1,0.2]
k_phi = [[] for _ in range(len(desAlpha))]
# Load simulation info
for i in range(len(desAlpha)):
    k_phi[i] = np.loadtxt(log_directory+'/k_phi-alpha='+str(desAlpha[i]))
    samples = len(k_phi[i])

fs = 35
lw = 3
sw = 1
tw = 40
lwsw = 200
ticks_size = 22
ms = 15

simTime = 500
labels = ['\\alpha=']

# === PLOT ===
t_axis = np.linspace(0, simTime, samples)
plt.figure(figsize=(16, 8))
for i in range(len(desAlpha)):
    plt.plot(t_axis, k_phi[i], label=r'$ %s $'%labels[0]+str(desAlpha[i]), linewidth=lw)

plt.xlabel('t (s)', fontsize=fs)
plt.ylabel('Cumulative loss function', fontsize=fs)
plt.legend(fontsize=fs * 0.7)
plt.grid(True)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.tight_layout()
plt.show()