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
spec = importlib.util.spec_from_file_location("module.tracker", class_path/'tracker.py')
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)

# Log path definition
plot_path = os.path.abspath('/home/andrea/Desktop/ros_simulation_ws/src/ipp_pkg/src/logs/plot')

# Initi Global Variables 
s_state = [0,0,0] # --> Agent Pose
t_pose = [0,0,0]
m_rx = [0,0,0,0]

def run_simulation(pub,auvID,auv,obs,Ts,Tf, auvNum):

    """Simulate the sensor platform and the moving target
    Input:  target : target initial state
            obs : list containing already initialized classes Tracker() (reproduce the local estimations)
            auv : list containing sensors state and methods for measurements
            pub : list containing the publishers
            cpf_control : already initialized class for CPF
            f : choosen geometry
            s_state : initial s state
    """
    global count1, s_state, t_pose, m_rx

    Hz = 1/(config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    dt = config.TIME_STEP*config.TIME_SCALER
    meas_table = []

    t_tdma = 0
    epsi = 0.01
    old_m = [0,0,0,0]


    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():

        

        if count1 == Hz:
            t_tdma += 1#TODO: SHOULD BE DIMENSIONED AFTER CHOOSING dt
            count1 = 0
            
            rospy.loginfo('AUV__'+str(auvID)+'__MEAS TABLE---------------------')
            rospy.loginfo(meas_table)

            if auvID*Ts == t_tdma:
                rospy.loginfo(str(auvID)+'AUV is transmitting - CHANNEL BUSY')
                # Perform measurement 
                [measure_, rel_bearing_, meas_pos] = auv.measureBearing(t_pose[0],t_pose[1],[s_state[0],s_state[1]],s_state[2])
                arr = [t,measure_,meas_pos[0],meas_pos[1]]
                pub[0].publish(np.array(arr,dtype=np.float32))
                if t_tdma == auvNum*Ts:
                    t_tdma = 0


                

        # TODO: PUT MEASUREMENTS PROCESSING HERE
        if sum(np.abs(m_rx[0:3]))-sum(np.abs(old_m[0:3]))>epsi:
        #if sum(m_rx) > 0:
            meas_table.append(m_rx)
        
        # Process the measurements and compute target state estimation if some conditions
        if len(meas_table) > 3:
            
            rospy.loginfo('AUV'+str(auvID)+'is MAKING AN ESTIMATION')
            time.sleep(50)
            obs.processMeasurement(meas_table)
            obs.propagate_estimation(t)

        # OPTIMIZATION OR OFFLINE PLANING MUST ACT HERE

        ctrl_cmd = np.array([auvID,0.1], dtype=np.float32)
        pub[2].publish(ctrl_cmd)
        '''rospy.loginfo('AUV STATE: (ID) and (POSE) and (TARGET STATE)')
        rospy.loginfo(auvID)
        rospy.loginfo(s_state)
        rospy.loginfo(t_pose)'''

        old_m = m_rx
        t += dt
        count1 += 1 
        if t_tdma >= Tf:
            t_tdma = 0
        rate.sleep()

def callback(data):
    
    global s_state
    s_state = data.data

def callback2(data):
    
    global t_pose
    t_pose = data.data

def callback3(data):
    
    global m_rx
    tmp = data.data
    m_rx = [tmp[0],tmp[1],tmp[2],tmp[3]]
    
def listener(auvID):

    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callback)
    rospy.Subscriber('target_state', numpy_msg(Floats), callback2)
    rospy.Subscriber('/'+str(auvID)+'/rx_meas', numpy_msg(Floats), callback3)
    
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
    pub_measurement = rospy.Publisher('/'+str(auvID)+'/tx_meas', numpy_msg(Floats), queue_size=100)
    pub_estimation = rospy.Publisher('estimation', numpy_msg(Floats), queue_size=10)
    pub_ctrl_cmd = rospy.Publisher('ctrl_cmd_'+str(auvID), numpy_msg(Floats),queue_size=10)
    pub.append(pub_measurement)
    pub.append(pub_estimation)  
    pub.append(pub_ctrl_cmd)

    # Initialize sensor and tracker object from costum class
    auv = sensor.Sensor(str(auvID),1,0,config.SIGMA_MEAS)
    obs = tracker.Tracker()

    # Init Communication protocol parameters (TDMA)
    Tf = config.Ts*auvNum
    # Start listener and simulation
    listener(auvID)
    run_simulation(pub,auvID,auv,obs,config.Ts,Tf,auvNum)

    rospy.spin()

if __name__ == '__main__':
    main()


