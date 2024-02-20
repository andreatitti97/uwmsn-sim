#!/usr/bin/env python
#Import basic system modules
import os
import time
import importlib.util, pathlib

# Import math modules
from math import pi, atan2
import numpy as np
from scipy import stats

#Import ROS modules
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# Import Costum classes
class_path = pathlib.Path(__file__).parent.resolve()
class_path = class_path/'Classes'
spec = importlib.util.spec_from_file_location("module.config", class_path/'config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
spec = importlib.util.spec_from_file_location("module.sensor", class_path/'sensor.py')
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)
# Log path definition
plot_path = os.path.abspath('/home/andrea/Desktop/ros_simulation_ws/src/ipp_pkg/src/logs/plot')

# Initi Global Variables 
s_pose = [0,0,0] # --> Agent Pose

def run_simulation(pub,auvID,auv):

    """Simulate the sensor platform and the moving target
    Input:  target : target initial state
            obs : list containing already initialized classes Tracker() (reproduce the local estimations)
            auv : list containing sensors state and methods for measurements
            pub : list containing the publishers
            cpf_control : already initialized class for CPF
            f : choosen geometry
            s_pose : initial s state
    """
    global count1, s_pose

    Hz = 1/(config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate
    Hz = 1
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    dt = config.TIME_STEP*config.TIME_SCALER

    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():


        
        a = np.array([auvID,0.1], dtype=np.float32)
        pub[1].publish(a)
        rospy.loginfo('AUV STATE: (ID) and (POSE)')
        rospy.loginfo(auvID)
        rospy.loginfo(s_pose)

        t += dt
        count1 += 1 

        rate.sleep()

def callback(data):
    
    global s_pose
    s_pose = data.data
    
def listener(auvID):

    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callback)
    
def main():

    # ROS INIT   
    namespace = rospy.get_namespace()
    params_path = namespace+'auv'
    # Get AUV ID and number of vehicles.
    auvID = rospy.get_param(params_path+'/auvID')
    auvNum = rospy.get_param(params_path+'/auvNum')
    # Node Init
    rospy.init_node('auv'+str(auvID))
    # Publishers init
    pub = []
    pub_estimation = rospy.Publisher('estimation', numpy_msg(Floats), queue_size=10)
    pub_ctrl_cmd = rospy.Publisher('ctrl_cmd_'+str(auvID), numpy_msg(Floats),queue_size=10)
    pub.append(pub_estimation)  
    pub.append(pub_ctrl_cmd)

    # Initialize sensor object from costum class
    auv = sensor.Sensor(str(auvID),1,0,config.SIGMA_MEAS)

    # Start listener and simulation
    listener(auvID)
    run_simulation(pub,auvID,auv)

    rospy.spin()

if __name__ == '__main__':
    main()


