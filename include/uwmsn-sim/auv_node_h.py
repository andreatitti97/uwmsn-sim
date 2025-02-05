#!/usr/bin/env python
#Import basic system modules
import os
import importlib.util, pathlib
import numpy as np
import time
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
spec = importlib.util.spec_from_file_location("module.sensor", class_path+'/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def updatePathRoutine(ID,s,headingRef,surgeRef,dt,DT):

    numSamples = 200
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

        tmp = []
        for i in range(len(rx)):
            
            tmp.append(np.sqrt((s[0]-rx[i])**2+(s[1]-ry[i])**2))
            
        idx = tmp.index(min(tmp))
        idx = 0
        idx_motion = 0
    else:
        idx_motion, idx = 0, 0
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