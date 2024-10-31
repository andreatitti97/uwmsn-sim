#!/usr/bin/env python
#Import basic system modules
import os
import importlib.util, pathlib
import numpy as np

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


def updatePathRoutine(ax,ay,waypoints,s,v_n,dt,DT):

    n_samples = 4
    # Initialized starting position
    a_i = [s[0],s[1]]
    t_i = s[2]

    # Compute new waypoints according to the given heading change
    for i in range(len(waypoints)):
        for j in range(n_samples): #more samples for better curve fitting()
            t_f = t_i+(waypoints[i]/int(DT/(DT/n_samples)))           
            ax.append(np.cos(t_f)*v_n*(DT/n_samples)+a_i[0])
            ay.append(np.sin(t_f)*v_n*(DT/n_samples)+a_i[1])
            a_i = [ax[-1],ay[-1]]
            t_i = t_f

    # Generate new path 
    path = planner.CubicSpline2D(ax, ay)
    [rx, ry, ryaw, rk, s, surge] = utils.calc_spline_course(path,dt)

    tmp = []
    for i in range(len(rx)):
        
        tmp.append(np.sqrt((s[0]-rx[i])**2+(s[1]-ry[i])**2))
        
    idx = tmp.index(min(tmp))
    idx_motion = 0

    return path, idx_motion, idx, rx, ry, ryaw, surge

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