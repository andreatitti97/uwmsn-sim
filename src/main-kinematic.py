#!/usr/bin/env python
#Import basic system modules
import os
import importlib.util, pathlib

# Import math modules
import numpy as np

#Import ROS modules
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# Environment: Define the relevant paths
'''pathlib: output is an object path (sum a string using '/')
os.path.dirname: output is a string with the path (sum strings using '+')'''

pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
log_path = pkg_directory+'/logs'
class_path = pkg_directory+'/src/Classes'

# Load the header file as a Python module 
header_file = pathlib.Path(__file__).parent.resolve()
header_file = os.path.dirname(header_file)
header_file = header_file+'/include'+'/uwmsn-sim'
spec = importlib.util.spec_from_file_location("module.header", header_file+'/main-kinematic_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

# Init lists for plot
# Target and AUVs
target_x_traj, target_y_traj, platform_x, platform_y = [], [], [], []
auv1_x, auv1_y, auv2_x, auv2_y,auv3_x,auv3_y,auv4_x,auv4_y  = [], [], [], [], [], [], [], []

ctrl_cmds = [0,0,0,0]

def run_simulation(target, auvNum, pub_s_state, pub_t_state):

    """Simulate the sensor platform and the moving header.target
    Input:  
            target : target initial state
            auvNum : list containing sensors state and methods for measurements
            pub_s_state : list containing the publishers for the vehicles states
            pub_t_state : list containing the publishers for the target state

    """
    global count1, ctrl_cmds

    # ROS simulation parameters
    Hz = 1/(header.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    dt = header.config.TIME_STEP*header.config.TIME_SCALER
 
    # Init AUVs position and orientation
    auvs_xy = np.zeros((4,2))
    auvs_theta = np.zeros(4) 
       
    
    for i in range(len(auvs_xy)):
        auvs_xy[i,0] = i*100
        auvs_xy[i,1] = 0

    rospy.loginfo('|---- KINEMATIC SIMULATION: Initial AUVs positions (m) --> %s',auvs_xy)
    rospy.loginfo('|---- KINEMATIC SIMULATION: Initial Target position (m) --> %s',[target.pose.x,target.pose.y,target.pose.theta])

    # Start listeners
    listener(auvNum)
    rospy.sleep(1)
    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():

        for i in range(auvNum):
            # Publish agents info
            a = np.array([auvs_xy[i,0],auvs_xy[i,1],auvs_theta[i]], dtype=np.float32)
            pub_s_state[i].publish(a)
            # Publish header.target info
            pub_t_state[i].publish(np.array([target.pose.x,target.pose.y,target.pose.theta], dtype=np.float32))

        # Move Agents
        for i in range(auvNum):
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
        if int(t) == (header.config.TIME_DURATION-1):
            rospy.on_shutdown(shutdown_cllbk)
            rospy.signal_shutdown('Simulation time limit reached')
      
        if count1 % Hz/10 == 0:
            '''ADD DEBUG PRINTS HERE'''
            rospy.loginfo('|---- KINEMATIC SIMULATION: Elapsed time (s) --> %s',t)
            rospy.loginfo('|---- KINEMATIC SIMULATION: Target groud truth (m) --> %s',[target.pose.x,target.pose.y])

        t += dt
        count1 += 1  
        rate.sleep()

def shutdown_cllbk():
    
    np.savetxt(log_path+'/target_x_traj.txt',target_x_traj)
    np.savetxt(log_path+'/target_y_traj.txt',target_y_traj)

    np.savetxt(log_path+'/auv1_x_ON.txt',auv1_x)
    np.savetxt(log_path+'/auv1_y_ON.txt',auv1_y)
    np.savetxt(log_path+'/auv2_x_ON.txt',auv2_x)
    np.savetxt(log_path+'/auv2_y_ON.txt',auv2_y)
    np.savetxt(log_path+'/auv3_x_ON.txt',auv3_x)
    np.savetxt(log_path+'/auv3_y_ON.txt',auv3_y)
    np.savetxt(log_path+'/auv4_x_ON.txt',auv4_x)
    np.savetxt(log_path+'/auv4_y_ON.txt',auv4_y)

    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('|---- %sKINEMATIC SIMULATION: Simulation data saved --> Shutting down ...%s',magenta,none)
    
def callback(data):
    global ctrl_cmds
    ctrl_cmd1, ctrl_cmd2, ctrl_cmd3, ctrl_cmd4 = 0,0,0,0
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

def listener(n_auv):
    
    for i in range(n_auv):
        rospy.Subscriber('/'+str(i+1)+'/ctrl_cmd_'+str(i+1), numpy_msg(Floats), callback)
    
def main():

    # ROS INIT   
    # Get AUV ID and number of vehicles.
    auvNum = rospy.get_param('/kinematic_sim/auvNum')
    # Node Init
    rospy.init_node('kinematic_sim') #log_level=rospy.DEBUG
    
    # Initialize publishers
    pub_t_state = []
    pub_s_state = []
    
    for i in range(auvNum):
        tmp1 = rospy.Publisher('/'+str(i+1)+'/vehicle_state_'+str(i+1), numpy_msg(Floats), queue_size=10)
        tmp2 = rospy.Publisher('/'+str(i+1)+'/target_state', numpy_msg(Floats), queue_size=10)
        pub_s_state.append(tmp1)
        pub_t_state.append(tmp2)
       
    # Initialization object header.target
    target_obj = header.target.Target()

    # Start listeners and run simulation
    run_simulation(target_obj, auvNum, pub_s_state, pub_t_state)  
    rospy.on_shutdown(shutdown_cllbk)
    rospy.spin()

if __name__ == '__main__':
    main()


