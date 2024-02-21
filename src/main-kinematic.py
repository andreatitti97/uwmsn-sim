#!/usr/bin/env python
#Import basic system modules
import os
import time
import importlib.util, pathlib
import matplotlib.pyplot as plt
# Import math modules
from math import pi, atan2
import numpy as np
from scipy import stats
#Import ROS modules
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#import matplotlib.pyplot as plt
# Import Costum classes
class_path = pathlib.Path(__file__).parent.resolve()

#class_path = os.path.abspath('/home/andrea/Desktop/ros_simulation_ws/src/ipp_pkg/src/Classes')
class_path = class_path/'Classes'

spec = importlib.util.spec_from_file_location("module.config", class_path/'config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
spec = importlib.util.spec_from_file_location("module.utils", class_path/'utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("module.tracker", class_path/'tracker.py')
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)

spec = importlib.util.spec_from_file_location("module.sensor", class_path/'sensor.py')
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)
spec = importlib.util.spec_from_file_location("module.target", class_path/'target.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)
spec = importlib.util.spec_from_file_location("module.cpf", class_path/'cpf.py')
cpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpf)
# PATH DEFINITON
plot_path = os.path.abspath('/home/andrea/Desktop/ros_simulation_ws/src/ipp_pkg/src/logs/plot')

# Init lists for plot
# Target and AUVs
target_x_traj, target_y_traj, platform_x, platform_y = [], [], [], []
auv1_x, auv1_y, auv2_x, auv2_y,auv3_x,auv3_y,auv4_x,auv4_y  = [], [], [], [], [], [], [], []
# Estimation Data
est1_x, est1_y, est2_x, est2_y,est3_x, est3_y,est_x, est_y, est_vx, est_vy = [],[], [], [], [], [], [], [], [], []
cov1, cov2, cov3, cov4, err_quad, cond_phi = [],[],[],[],[],[]

ctrl_cmd1,ctrl_cmd2,ctrl_cmd3,ctrl_cmd4 = 0,0,0,0
ctrl_cmds = [0,0,0,0]

def compute_cost(phi,len_y):

    tmp_phi = np.zeros((len_y,4))
    for i in range(len_y):
        row = phi[i]
        tmp_phi[i,:] = [row[0],row[1],row[2],row[3]]
    PHI = np.dot(np.transpose(tmp_phi[:,0:2]),tmp_phi[:,0:2])
    cost = np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)
    return cost


def computeCov(y,phi):

    # Compute Covariance of the target state
    R = np.zeros((len(y),len(y))) #matrice diagonale perchè errori sulle singole misure indipendenti tra loro            
    for i in range(len(y)): 
        for j in range(len(y)):
            if i == j:
                R[i,j] = (config.SIGMA_MEAS)
            else:
                R[i,j] = 0 
    if count1 > 0 and len(phi)>=4:
        a = config.SIGMA_MEAS
        cov = np.linalg.inv(np.dot(np.dot(np.transpose(phi),np.linalg.inv(a*np.identity(len(y)))),phi))
    else: 
        cov = np.zeros((4,4))
    return cov

def updatePathRoutine(rx,ry,s_pose):

    tmp = []
    for i in range(len(rx)):
        
        tmp.append(np.sqrt((s_pose[0]-rx[i])**2+(s_pose[1]-ry[i])**2))
        
    idx = tmp.index(min(tmp))
  
    idx_motion = 0

    return idx_motion, idx

def sig(x):
    
    alpha = -0.003
    gamma = config.d
    return 1/(1+np.e**(alpha*(gamma-x)))

def run_simulation(target, auvNum, pub_s_state, pub_t_state):

    """Simulate the sensor platform and the moving target
    Input:  target : target initial state
            obs : list containing already initialized classes Tracker() (reproduce the local estimations)
            auv : list containing sensors state and methods for measurements
            pub_t_state : list containing the publishers
            cpf_control : already initialized class for CPF
            f : choosen geometry
            s_pose : initial s state
    """
    global count1, ctrl_cmds, ctrl_cmd1, ctrl_cmd2, ctrl_cmd3, ctrl_cmd4
    propagation = False
    N = auvNum
    Hz = 1/(config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate
    
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    dt = config.TIME_STEP*config.TIME_SCALER

 
    # Init AUVs position and orientation according to given formation
    auvs_xy = np.zeros((4,2))
    auvs_theta = np.zeros(4) 
    # Initialize nominal vel for the CPF algorithm
    v_n = config.AUV_VEL   
    listener(auvNum)
    rospy.sleep(1)
    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():

        #rospy.loginfo('SIMULATION TIME(s)')
        #rospy.loginfo(t)

        for i in range(N):
            # Publish agents info
            a = np.array([auvs_xy[i,0],auvs_xy[i,1],auvs_theta[i]], dtype=np.float32)
            pub_s_state[i].publish(a)
            # Publish target info
            pub_t_state[i].publish(np.array([target.pose.x,target.pose.y,target.pose.theta], dtype=np.float32))

        # Move Agents
        for i in range(N):
            auvs_xy[i,0] = auvs_xy[i,0]+ctrl_cmds[i]
            auvs_xy[i,1] = auvs_xy[i,1]+ctrl_cmds[i]
            auvs_theta[i] = auvs_theta[i]+ctrl_cmds[i]

        # Move Target
        target.move_target(dt)

        #################################################################################################################
        ##################### SAVE THE POSITIONS OF TEAM REFERENCE/AGENTS/TARGET/ STATE FOR PLOT ########################
        
        auv1_x.append(auvs_xy[0,0])
        auv1_y.append(auvs_xy[0,1])
        if auvNum>1:
            auv2_x.append(auvs_xy[1,0])
            auv2_y.append(auvs_xy[1,1])
        if auvNum>2:
            auv3_x.append(auvs_xy[2,0])
            auv3_y.append(auvs_xy[2,1])
            auv4_x.append(auvs_xy[3,0])
            auv4_y.append(auvs_xy[3,1])
        target_x_traj.append(target.pose.x)
        target_y_traj.append(target.pose.y)         
        ##################################################################################################################
        #  Stop simulation and save data to .txt files ###################################################################
        if int(t) == (config.TIME_DURATION-1):
            rospy.loginfo('saving data for plot')
            np.savetxt(plot_path+'/target_x_traj.txt',target_x_traj)
            np.savetxt(plot_path+'/target_y_traj.txt',target_y_traj)
            if config.OPTIMIZATION_ON == True:
                np.savetxt(plot_path+'/est4_x_ON.txt',est_x)
                np.savetxt(plot_path+'/est4_y_ON.txt',est_y)
                np.savetxt(plot_path+'/err_quad_ON.txt',err_quad)
                np.savetxt(plot_path+'/x_platform_ON.txt',platform_x)
                np.savetxt(plot_path+'/y_platform_ON.txt',platform_y)
                np.savetxt(plot_path+'/auv1_x_ON.txt',auv1_x)
                np.savetxt(plot_path+'/auv1_y_ON.txt',auv1_y)
                np.savetxt(plot_path+'/auv2_x_ON.txt',auv2_x)
                np.savetxt(plot_path+'/auv2_y_ON.txt',auv2_y)
                np.savetxt(plot_path+'/auv3_x_ON.txt',auv3_x)
                np.savetxt(plot_path+'/auv3_y_ON.txt',auv3_y)
                np.savetxt(plot_path+'/auv4_x_ON.txt',auv4_x)
                np.savetxt(plot_path+'/auv4_y_ON.txt',auv4_y)
                np.savetxt(plot_path+'/cov1_ON.txt',cov1)
                np.savetxt(plot_path+'/cov2_ON.txt',cov2)
                np.savetxt(plot_path+'/cov3_ON.txt',cov3)
                np.savetxt(plot_path+'/cov4_ON.txt',cov4)
                np.savetxt(plot_path+'/vx_ON.txt',est_vx)
                np.savetxt(plot_path+'/vy_ON.txt',est_vy)
                np.savetxt(plot_path+'/cond_ON',cond_phi)
   
            else:
                np.savetxt(plot_path+'/est4_x_OFF.txt',est_x)
                np.savetxt(plot_path+'/est4_y_OFF.txt',est_y)
                np.savetxt(plot_path+'/err_quad_OFF.txt',err_quad)
                np.savetxt(plot_path+'/x_platform_OFF.txt',platform_x)
                np.savetxt(plot_path+'/y_platform_OFF.txt',platform_y)
                np.savetxt(plot_path+'/auv1_x_OFF.txt',auv1_x)
                np.savetxt(plot_path+'/auv1_y_OFF.txt',auv1_y)
                np.savetxt(plot_path+'/auv2_x_OFF.txt',auv2_x)
                np.savetxt(plot_path+'/auv2_y_OFF.txt',auv2_y)
                np.savetxt(plot_path+'/auv3_x_OFF.txt',auv3_x)
                np.savetxt(plot_path+'/auv3_y_OFF.txt',auv3_y)
                np.savetxt(plot_path+'/auv4_x_OFF.txt',auv4_x)
                np.savetxt(plot_path+'/auv4_y_OFF.txt',auv4_y)
                np.savetxt(plot_path+'/cov1_OFF.txt',cov1)
                np.savetxt(plot_path+'/cov2_OFF.txt',cov2)
                np.savetxt(plot_path+'/cov3_OFF.txt',cov3)
                np.savetxt(plot_path+'/cov4_OFF.txt',cov4)
                np.savetxt(plot_path+'/vx_OFF.txt',est_vx)
                np.savetxt(plot_path+'/vy_OFF.txt',est_vy)
                np.savetxt(plot_path+'/cond_OFF',cond_phi)
                np.savetxt(plot_path+'/cond_ON',cond_phi)
              
        t += dt
        count1 += 1  
        rate.sleep()

def callback(data):
    global ctrl_cmds, ctrl_cmd1, ctrl_cmd2, ctrl_cmd3, ctrl_cmd4
    
    tmp = data.data

    if int(tmp[0]) == 1:
        ctrl_cmd1 = tmp[1]
    elif int(tmp[0]) == 2:
        ctrl_cmd2 = tmp[1]
    elif int(tmp[0]) == 3:
        ctrl_cmd3 = tmp[1]
    elif int(tmp[0]) == 4:
        ctrl_cmd4 = tmp[1]
    ctrl_cmds = [ctrl_cmd1,ctrl_cmd2,ctrl_cmd3,ctrl_cmd4]
    
    #ctrl_cmds[i] = tmp[1]
    
def listener(n_auv):
    
    for i in range(n_auv):
        rospy.Subscriber('/'+str(i+1)+'/ctrl_cmd_'+str(i+1), numpy_msg(Floats), callback)
    
def main():

    # ROS INIT   
    namespace = rospy.get_namespace()
    params_path = namespace
    # Get AUV ID and number of vehicles.
    auvNum = rospy.get_param('/kinematic_sim/auvNum')
    # Node Init
    rospy.init_node('kinematic_sim')

    # Initialize publishers
    pub_t_state = []
    pub_s_state = []
    
    for i in range(auvNum):
        tmp1 = rospy.Publisher('/'+str(i+1)+'/vehicle_state_'+str(i+1), numpy_msg(Floats), queue_size=10)
        tmp2 = rospy.Publisher('/'+str(i+1)+'/target_state', numpy_msg(Floats), queue_size=10)
        pub_s_state.append(tmp1)
        pub_t_state.append(tmp2)
       

    # Initialization object target
    target_obj = target.Target()

    # Start listeners and run simulation
    
    run_simulation(target_obj, auvNum, pub_s_state, pub_t_state)
    rospy.spin()

if __name__ == '__main__':
    main()


