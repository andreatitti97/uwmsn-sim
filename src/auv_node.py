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
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
header_file = pkg_directory+'/include'+'/uwmsn-sim'
log_path = pkg_directory+'/logs'

# Load the header file as a Python module 
spec = importlib.util.spec_from_file_location("module.header", header_file+'/auv_node_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

# Init empy lists for saving simulation data
x_hat_1,x_hat_2,x_hat_3,x_hat_4 = [], [], [], []
cov1, cov2, cov3, cov4, err = [], [], [], [], []

# Init Global Variables 
s_state = [0,0,0] # --> Agent Pose
t_pose = [0,0,0] # --> Target ground truth
m_rx = [0,0,0,0] # --> received measurament

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
                R[i,j] = (header.config.SIGMA_MEAS)
            else:
                R[i,j] = 0 

    a = header.config.SIGMA_MEAS
    cov = np.linalg.inv(np.dot(np.dot(np.transpose(phi),np.linalg.inv(a*np.identity(len(y)))),phi))
        
    return cov

def run_auv_node(pub,auv,obs,Ts,Tf, auvNum):

    """Simulate the header.sensor platform and the moving target
    Input:              
            pub : list containing the publishers
            auv : list containing sensors state and methods for measurements
            obs : object containing already initialized classes Tracker() (reproduce the local estimations)
            Ts  : slot time of the TDMA protocol
            Tf  : frame time of the TDMA protocol
            auvNum : number ora AUVs
    """
    global count1, s_state, t_pose, m_rx, auvID

    # ROS simulation parameters
    t_scaler = header.config.TIME_SCALER

    Hz = 1/(header.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)

 
    # Colors for prints
    blue = "\033[1;34m"
    cyan = "\033[0;36m"
    none = "\033[0m"
    listener(auvID) #start the listeners

    # Init time variables and counters and lists
    t, count1, t_tdma = 0,0,0
    dt = header.config.TIME_STEP*t_scaler
    meas_table = []
    local_measures = []
    old_m = [0,0,0,0]
    thresh = 1
    # Start listeners
    listener(auvID)

    rospy.sleep(1)
    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():

        if (count1 % (Hz/t_scaler))== 0:
            t_tdma += 1#TODO: SHOULD BE DIMENSIONED AFTER CHOOSING dt
            
            if auvID*Ts == t_tdma:
                
                rospy.loginfo('%s|---- AUV '+str(auvID)+': Transmitting measurements at time %s --> Channel Busy%s',cyan,t,none)
                
                # Perform measurement 
                [measure_, rel_bearing_, meas_pos] = auv.measureBearing(t_pose[0],t_pose[1],[s_state[0],s_state[1]],s_state[2])
                arr = [t,measure_,meas_pos[0],meas_pos[1]]
                local_measures.append(arr)
                for i in range(len(local_measures)):
                    pub[0].publish(np.array(local_measures[i],dtype=np.float32))
                local_measures = []
                if t_tdma == auvNum*Ts:
                    t_tdma = 0
            else:#for making measurements also outside the given timeslot 
                [measure_, rel_bearing_, meas_pos] = auv.measureBearing(t_pose[0],t_pose[1],[s_state[0],s_state[1]],s_state[2])
                arr = [t,measure_,meas_pos[0],meas_pos[1]]
                local_measures.append(arr)

        # TODO: PUT MEASUREMENTS PROCESSING HERE
        if m_rx[0] - old_m[0] > 1:#check if the measurement is new

            meas_table.append(m_rx)
            rospy.logdebug('|---- AUV '+str(auvID)+': Measuraments Table [t,y,p_sx,p_sy]--> %s',meas_table)

        # Process the measurements and compute target state estimation if some conditions
        if len(meas_table) > auvNum: #just to be sure there are enough measurements avoiding sing matrix

            obs.processMeasurement(meas_table)
            phi,y = obs.regressor #curr_est at this point is not used!
            meas_table = []

        # if good conditioning do estimation 
            if compute_cost(phi,len(y)) >= thresh:
                obs.propagate_estimation(t) #you can now propagate
                curr_est = obs.state
                cov = computeCov(y,phi)
                
                rospy.logout('%s|---- AUV '+str(auvID)+': Target state Estimation [m,m/s] --> %s%s',blue,curr_est,none)
                # Computte the tracking error
                err_x = (t_pose[0] - curr_est[0,0])
                err_y = (t_pose[1] - curr_est[1,0])
                e = np.sqrt(err_x**2+err_y**2)
                # Save Estimation Data #####################################################################################
                x_hat_1.append(curr_est[0,0])
                x_hat_2.append(curr_est[1,0])
                x_hat_3.append(curr_est[2,0])
                x_hat_4.append(curr_est[3,0])
                err.append(e)
                # Save Covariance associated 
                cov1.append(cov[0,0])
                cov2.append(cov[1,1])
                cov3.append(cov[2,2])
                cov4.append(cov[3,3])
                ############################################################################################################
            
        #TODO: OPTIMIZATION OR OFFLINE PLANING MUST ACT HERE!

        ctrl_cmd = np.array([auvID,0.1], dtype=np.float32)
        pub[2].publish(ctrl_cmd)

        if int(t) == (header.config.TIME_DURATION-1):
            rospy.on_shutdown(shutdown_cllbk)
            rospy.signal_shutdown('Simulation time limit reached')

        old_m = m_rx
        t += dt
        count1 += 1 
        if t_tdma >= Tf:
            t_tdma = 0
        rate.sleep()

def shutdown_cllbk():
    global auvID
   
    '''PUT DATA SAVING HERE'''
    np.savetxt(log_path+'/'+str(auvID)+'-x_hat_1.txt',x_hat_1)
    np.savetxt(log_path+'/'+str(auvID)+'-x_hat_2.txt',x_hat_2)
    np.savetxt(log_path+'/'+str(auvID)+'-x_hat_3.txt',x_hat_3)
    np.savetxt(log_path+'/'+str(auvID)+'-x_hat_4.txt',x_hat_4)
    np.savetxt(log_path+'/'+str(auvID)+'-err.txt',err)

    np.savetxt(log_path+'/'+str(auvID)+'-cov1.txt',cov1)
    np.savetxt(log_path+'/'+str(auvID)+'-cov2.txt',cov2)
    np.savetxt(log_path+'/'+str(auvID)+'-cov3.txt',cov3)
    np.savetxt(log_path+'/'+str(auvID)+'-cov4.txt',cov4)
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- AUV '+str(auvID)+': Simulation data saved --> Shutting down ...%s',magenta,none)

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
    global auvID
    auvID = rospy.get_param(params_path+'/auvID')
    auvNum = rospy.get_param(params_path+'/auvNum')

    # Node Init
    rospy.init_node('auv'+str(auvID)) #TO ADD debug prints --> log_level=rospy.DEBUG

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
    run_auv_node(pub,auv,obs,header.config.Ts,Tf,auvNum)
    rospy.on_shutdown(shutdown_cllbk)
    rospy.spin()

if __name__ == '__main__':
    main()


