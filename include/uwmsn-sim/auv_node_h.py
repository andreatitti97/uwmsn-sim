#!/usr/bin/env python
#Import basic system modules
import os
import importlib.util, pathlib
import numpy as np
import time
# Modules for DTW
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import Costum classes
class_path = pathlib.Path(__file__).parent.resolve()
class_path = os.path.dirname(os.path.dirname(class_path))
class_path = class_path+'/src'+'/Classes'

spec = importlib.util.spec_from_file_location("module.config", class_path+'/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
spec = importlib.util.spec_from_file_location("module.sensor", class_path+'/sensor.py')
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)
spec = importlib.util.spec_from_file_location("module.tracker", class_path+'/tracker.py')
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)
spec = importlib.util.spec_from_file_location("module.planner", class_path+'/spline_planner.py')
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
spec = importlib.util.spec_from_file_location("module.etc", class_path+'/etc.py')
etc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(etc)
spec = importlib.util.spec_from_file_location("module.sensor", class_path+'/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def updatePathRoutine(ID,s,headingRef,surgeRef,dt,DT):

    numSamples = 200
    idx_motion, idx = 0, 0
    ax, ay = [], []
    # Initialized starting position
    a_i = [s[0],s[1]]
    h_i = s[2]

    # Compute new headingRef according to the given heading change
    for i in range(config.H):
        for _ in range(numSamples): #more samples for better curve fitting()
            h_f = h_i+(headingRef[i]/int(DT/(DT/numSamples)))     
            if surgeRef[i] > 10e-4:      
               
                a_i[0] = a_i[0]+np.cos(h_f)*surgeRef[i]*(DT/numSamples)
                a_i[1] = a_i[1]+np.sin(h_f)*surgeRef[i]*(DT/numSamples)
                ax.append(a_i[0])
                ay.append(a_i[1])
            
            h_i = h_f

    # Generate new path 
    if len(ax) > 1 and len(ay) > 1:
        path = planner.CubicSpline2D(ax, ay)
        [rx, ry, ryaw, rk, spline, surge] = utils.calc_spline_course(path,dt)
    else:
        rx, ry, ryaw = [s[0]], [s[1]], [s[2]]
        path = None

    return path, idx_motion, idx, rx, ry, ryaw

def orderByTimestamp(data_list):
    """
    Sorts a list of lists based on the timestamp in each sublist.

    Parameters:
    - data_list (list of lists): Each sublist has a timestamp as the first element.

    Returns:
    - list of lists: Sorted list based on timestamps.
    """
    # Sort data_list based on the first element (timestamp) of each sublist
    return sorted(data_list, key=lambda x: x[0])


# Compute DTW (Dynamic Time Warping) distance between two 2D control sequences
def compute_dtw(U_k, U_k_1):
    """
    Compute the DTW distance between two control sequences U_k and U_k_1.
    Each sequence is a 2D array (time steps with surge and sway).
    """
    # Ensure U_k and U_k_1 are 2D arrays
    U_k = np.array(U_k).reshape(-1, 2)  # Ensure each control point is 2D
    U_k_1 = np.array(U_k_1).reshape(-1, 2)

    # Convert the control sequences into a list of tuples for DTW
    U_k_tuples = [tuple(x) for x in U_k]  # Each control point is a 2D tuple
    U_k1_tuples = [tuple(x) for x in U_k_1]
    
    distance, _ = fastdtw(U_k_tuples, U_k1_tuples, dist=euclidean)
    return distance

# Decide whether to retransmit based on the DTW distance and adaptive threshold
def retransmit_decision(U_k, U_k_1, base_threshold, packet_loss_factor, latency_factor, consensus_error, alpha):
    """
    Decide whether to retransmit based on the discrepancy between control sequences (DTW) 
    and the adaptive threshold considering packet loss, latency, and consensus error.
    """
    # Compute the DTW distance between U_k and U_k_1
    dtw_distance = compute_dtw(U_k, U_k_1)
    
    # Update the adaptive threshold based on consensus error
    adaptive_threshold = base_threshold #- alpha * consensus_error

    retransmit = dtw_distance >= adaptive_threshold
    return retransmit, adaptive_threshold, dtw_distance