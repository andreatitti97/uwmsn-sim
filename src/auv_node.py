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

# Load the header file as a Python module 
header_file = pathlib.Path(__file__).parent.resolve()
header_file = os.path.dirname(header_file)
header_file = header_file+'/include'+'/uwmsn-sim'
spec = importlib.util.spec_from_file_location("module.header", header_file+'/auv_node_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

# Init Global Variables 
s_state = [0,0,0] # --> Agent Pose
t_pose = [0,0,0] # --> Target ground truth
m_rx = [0,0,0,0] # --> received measurament

def run_simulation(pub,auvID,auv,obs,Ts,Tf, auvNum):

    """Simulate the header.sensor platform and the moving target
    Input:  target : target initial state
            obs : list containing already initialized classes Tracker() (reproduce the local estimations)
            auv : list containing sensors state and methods for measurements
            pub : list containing the publishers
            cpf_control : already initialized class for CPF
            f : choosen geometry
            s_state : initial s state
    """
    global count1, s_state, t_pose, m_rx

    Hz = 1/(header.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate
    rate = rospy.Rate(Hz)
    listener(auvID) #start the listeners

    # Init time variables and counters and lists
    t, count1, t_tdma = 0,0,0
    dt = header.config.TIME_STEP*header.config.TIME_SCALER
    meas_table = []
    old_m = [0,0,0,0]

    # Start listeners
    listener(auvID)
    rate.sleep()

    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():

        if count1 == Hz:
            t_tdma += 1#TODO: SHOULD BE DIMENSIONED AFTER CHOOSING dt
            count1 = 0

            if auvID*Ts == t_tdma:
                rospy.loginfo(str(auvID)+'AUV is transmitting - CHANNEL BUSY')
                rospy.loginfo(t)
                # Perform measurement 
                [measure_, rel_bearing_, meas_pos] = auv.measureBearing(t_pose[0],t_pose[1],[s_state[0],s_state[1]],s_state[2])
                arr = [t,measure_,meas_pos[0],meas_pos[1]]
                pub[0].publish(np.array(arr,dtype=np.float32))
                if t_tdma == auvNum*Ts:
                    t_tdma = 0

        # TODO: PUT MEASUREMENTS PROCESSING HERE
        if m_rx[0] - old_m[0] > 1:#check if the measurement is new

            meas_table.append(m_rx)
            rospy.loginfo('AUV__'+str(auvID)+'__MEAS TABLE---------------------')
            rospy.loginfo(meas_table)
        
        # Process the measurements and compute target state estimation if some conditions
        if len(meas_table) > 5:
            
            #rospy.loginfo('AUV'+str(auvID)+'is MAKING AN ESTIMATION')
            obs.processMeasurement(meas_table)
            obs.propagate_estimation(t)
            curr_est,phy,y = obs.state
            if count1 % 100 == 0:
                rospy.loginfo('----------------------------------------------Target STATE ESTIMATION from AUV'+str(auvID))
                rospy.loginfo(curr_est)
            

        # OPTIMIZATION OR OFFLINE PLANING MUST ACT HERE

        ctrl_cmd = np.array([auvID,0.0], dtype=np.float32)
        pub[2].publish(ctrl_cmd)

        if int(t) == (header.config.TIME_DURATION-1):
            rospy.signal_shutdown('Simulation time limit reached')

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

    # Initialize header.sensor and header.tracker object from costum class
    auv = header.sensor.Sensor(str(auvID),1,0,header.config.SIGMA_MEAS)
    obs = header.tracker.Tracker()

    # Init Communication protocol parameters (TDMA)
    Tf = header.config.Ts*auvNum
    # Start simulation
    
    run_simulation(pub,auvID,auv,obs,header.config.Ts,Tf,auvNum)

    rospy.spin()

if __name__ == '__main__':
    main()


