#!/usr/bin/env python
#Import basic system modules
import os, importlib.util, pathlib
# Import math modules
import numpy as np
from math import atan2
#Import ROS modules
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# Environment: Define the relevant paths
'''pathlib: output is an object path (sum a string using '/')
os.path.dirname: output is a string with the path (sum strings using '+')'''

pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
log_path = os.path.dirname(pkg_directory)+'/logs'
class_path = pkg_directory+'/src/Classes'

# Load the header file as a Python module 
header_file = pathlib.Path(__file__).parent.resolve()
header_file = os.path.dirname(header_file)
header_file = header_file+'/include'+'/uwmsn-sim'
spec = importlib.util.spec_from_file_location("module.header", header_file+'/main-kinematic_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

# Init lists for plot
target_x_traj, target_y_traj, platform_x, platform_y = [], [], [], []
auv1_x, auv1_y, auv2_x, auv2_y,auv3_x,auv3_y,auv4_x,auv4_y  = [], [], [], [], [], [], [], []
# Init global variables for callbacks
H = header.config.H
path1, path2, path3, path4, paths = [], [], [], [], []
paths = [None,None,None,None]
for i in range(H):
    path1.append(None)
    path2.append(None)
    path3.append(None)
    path4.append(None)

def run_simulation(target, auvNum, pub_s_state, pub_t_state, pub_init_opt):

    """Simulate the sensor platform and the moving target
    Input:  
            target : target initial state
            auvNum : list containing sensors state and methods for measurements
            pub_s_state : list containing the publishers for the vehicles states
            pub_t_state : list containing the publishers for the target state

    """
    global count1, paths

    # ROS simulation parameters
    t_scaler = header.config.TIME_SCALER
    Hz = 1/(header.config.TIME_STEP) 
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    dt = header.config.TIME_STEP*t_scaler
 
    # Init AUVs position and orientation
    auvs_xy = header.config.AUV_XY
    msg, sensors, meas_table = [], [], []

    for i in range(len(auvs_xy)):

        msg.append(auvs_xy[i,0])
        msg.append(auvs_xy[i,1])
        msg.append(auvs_xy[i,2])
        sensors.append(header.sensor.Sensor(str(i),1,0,0.000))
    
    for i in range(int(auvNum)):
        
        [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.pose.x,target.pose.y,
                                                                       [auvs_xy[i,0],auvs_xy[i,1]],
                                                                       auvs_xy[i,2])
        arr = [measure_,meas_pos[0],meas_pos[1]]
        meas_table.append(arr)

    estimator = header.estimator_module.Estimation()
    estimator.computeState(meas_table)
    rospy.loginfo('|---- OPTIMIZATION TIME WINDOW (s) --> %s',header.config.DT)
    rospy.loginfo('|---- KINEMATIC SIMULATION: Initial AUVs positions (m) --> %s',auvs_xy)
    rospy.loginfo('|---- KINEMATIC SIMULATION: Initial Target position (m) --> %s',
                    [target.pose.x,target.pose.y,target.pose.theta])
    rospy.loginfo('|---- INITIAL OBJECTIVE(CONDITION) --> %s',header.utils.compute_cost(estimator.phi))
    rospy.sleep(1)

    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():
        if count1 < Hz: #to be sure that the initalization setup is shared among nodes
            pub_init_opt[0].publish(np.array(msg,dtype=np.float32)) #ONE TIME PUBLISHER
            tmp = []
            for i in range(auvNum):
                tmp.append(np.sqrt((target.pose.y-auvs_xy[i,1])**2+(target.pose.x-auvs_xy[i,0])**2))
            pub_init_opt[1].publish(np.array(tmp,dtype=np.float32))
            
        for i in range(auvNum):
            # Publish agents info
            a = np.array([auvs_xy[i,0],auvs_xy[i,1],auvs_xy[i,2]], dtype=np.float32)
            pub_s_state[i].publish(a)
            # Publish target info
            tmp = []
            pub_t_state[i].publish(np.array([target.pose.x,target.pose.y,target.pose.theta],
                                            dtype=np.float32))

        # Move Agents
        for i in range(auvNum):
            paths = [path1,path2,path3,path4]
            tmp = paths[i]

            if tmp[0] != None:

                auvs_xy[i,0] = tmp[0]
                auvs_xy[i,1] = tmp[1]
                auvs_xy[i,2] = tmp[2]

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
        if int(t) == (header.config.TIME_DURATION-1):
            rospy.on_shutdown(shutdown_cllbk)
            rospy.signal_shutdown('Simulation time limit reached')
      
        if count1 % Hz == 0:
            '''ADD DEBUG PRINTS HERE'''
            rospy.loginfo('|---- KINEMATIC SIMULATION: Elapsed time (s) --> %s',t)
            rospy.loginfo('|---- KINEMATIC SIMULATION: Target groud truth (m) --> %s',
                            [target.pose.x,target.pose.y])

        t += dt
        count1 += 1  
        rate.sleep()

def shutdown_cllbk():
    
    np.savetxt(log_path+'/target_x_traj.txt',target_x_traj)
    np.savetxt(log_path+'/target_y_traj.txt',target_y_traj)
    np.savetxt(log_path+'/auv1_x_traj.txt',auv1_x)
    np.savetxt(log_path+'/auv1_y_traj.txt',auv1_y)
    np.savetxt(log_path+'/auv2_x_traj.txt',auv2_x)
    np.savetxt(log_path+'/auv2_y_traj.txt',auv2_y)
    np.savetxt(log_path+'/auv3_x_traj.txt',auv3_x)
    np.savetxt(log_path+'/auv3_y_traj.txt',auv3_y)
    np.savetxt(log_path+'/auv4_x_traj.txt',auv4_x)
    np.savetxt(log_path+'/auv4_y_traj.txt',auv4_y)
    

    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('|---- %sKINEMATIC SIMULATION: Simulation data saved --> Shutting down ...%s',magenta,none)
    rospy.loginfo('|---- Simulation Info: alhpa %s gama %s NL %s DThresh %s desired range %s',
                  header.config.alpha_w,header.config.gamma_w, header.config.NL,
                  header.config.DThresh, header.config.RANGE_TO_TARGET)
def callback1(data):
    global paths,path1
    
    tmp = data.data   
    path1 = [tmp[1],tmp[2],tmp[3]]

def callback2(data):
    global paths, path2    

    tmp = data.data
    path2 = [tmp[1],tmp[2],tmp[3]]

def callback3(data):
    global paths, path3

    tmp = data.data
    path3 = [tmp[1],tmp[2],tmp[3]]

def callback4(data):
    global paths, path4    

    tmp = data.data
    path4 = [tmp[1],tmp[2],tmp[3]]

def listener(n_auv):
    callback_list = [callback1,callback2,callback3,callback4]
    for i in range(n_auv):
        rospy.Subscriber('/'+str(i+1)+'/ctrl_cmd_'+str(i+1), numpy_msg(Floats), callback_list[i])
    
def main():

    # ROS INIT   
    # Get AUV ID and number of vehicles.
    auvNum = rospy.get_param('/kinematic_sim/auvNum')
    # Node Init
    rospy.init_node('kinematic_sim') #log_level=rospy.DEBUG
    
    # Initialize publishers
    pub_t_state, pub_s_state ,pub_init_opt= [], [], []
    tmp1 = rospy.Publisher('/init_opt1', numpy_msg(Floats), queue_size=100)
    tmp2 = rospy.Publisher('/init_opt2', numpy_msg(Floats), queue_size=100)
    pub_init_opt.append(tmp1)
    pub_init_opt.append(tmp2)
    for i in range(auvNum):
        tmp1 = rospy.Publisher('/'+str(i+1)+'/vehicle_state_'+str(i+1), numpy_msg(Floats), queue_size=100)
        tmp2 = rospy.Publisher('/'+str(i+1)+'/target_state', numpy_msg(Floats), queue_size=100)

        pub_s_state.append(tmp1)
        pub_t_state.append(tmp2)

    # Initialization object header.target
    target_obj = header.target.Target(1)

    # Save a logfile with simulation settings
    sim_info = [auvNum, header.config.TIME_DURATION, header.config.Ts]
    ctrl_set = header.config.ctrl_cmd
    np.savetxt(log_path+'/sim_info.txt',sim_info)
    np.savetxt(log_path+'/ctrl_set.txt',ctrl_set)

    # Start listeners and run simulation
    listener(auvNum)
    run_simulation(target_obj, auvNum, pub_s_state, pub_t_state, pub_init_opt)  
    rospy.on_shutdown(shutdown_cllbk)
    rospy.spin()

if __name__ == '__main__':
    main()


