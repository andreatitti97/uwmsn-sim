#!/usr/bin/env python
#Import basic system modules
import os, importlib.util, pathlib
# Import math modules
import numpy as np
#Import ROS modules
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from uwmsn_msgs.msg import Matrix

# Environment: Define the relevant paths
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
header_file = pkg_directory+'/include'+'/uwmsn-sim'
log_path = os.path.dirname(pkg_directory)+'/logs'

# Load the h file as a Python module 
spec = importlib.util.spec_from_file_location("module.header", header_file+'/auv_node_h.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
splinePlanner = h.planner

# Init Global Variables for ROS callbacks
AUV_XY = h.config.AUV_XY
targetNumSim = h.config.targetNum

# Sensor state [px,py,yaw]
senPose = [0,0,0] 

# Target ground truth [px,py,yaw,exist=bool,label]
targetsData = [[0,0,0,0] for _ in range(targetNumSim)] 

# Received measurament [[t1,meas,ps_x_t1,ps_y_t1,label],...,[t1,meas,ps_x_tN,ps_y_tN,label]]
measRx = [[0,0,0,0,0] for _ in range(targetNumSim)]# everything initialized to 0

# Ctrl policy [ps_x, ps_y, theta_s, r_1, ... r_H, u_1, ... u_H]
ctrlPolicy = [0 for _ in range(len(senPose)+(h.config.H+1)*2)]

# Empty list for plots
heading, surge_vel = [], []
trackErr = [[] for _ in range(targetNumSim)]


def run_auv_node(pub,auv,obs,Ts,Tf,auvNum):

    """Simulate the h.sensor platform and the moving target
    Input:              
            pub : list containing the publishers
            auv : list containing sensors state and methods for measurements
            obs : object containing already initialized classes Tracker() (reproduce the local estimations)
            Ts  : slot time of the TDMA protocol
            Tf  : frame time of the TDMA protocol
            auvNum : number ora AUVs
    """
    global senPose, targetsData, measRx, auvID, ctrlPolicy

    # ROS simulation parameters
    t_scaler, TM = h.config.TIME_SCALER, h.config.TM

    Hz = 1/(h.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)

    # Colors for prints
    blue = "\033[1;34m"
    cyan = "\033[0;36m"
    none = "\033[0m"

    # Load simulation params from config file
    dt, thresh = h.config.TIME_STEP*t_scaler, h.config.k_phi_thresh
    AUV_failure, P_min = h.config.AUV_failure, h.config.P_min
    DT = h.config.Ts*auvNum*2
    targetNum = len(obs)

    # Init time variables and counters and lists
    t, count1, clkTdma, clkSmpl, tol = 0,0,0,0,1
    measTx, msgTx = [], []
    measTable = [[] for _ in range(len(targetsData))]
    measRxOld = [[0,0,0,0,0] for _ in range(targetNum)]
    path = None

    # Start listeners and init waypoints data structure
    ax, ay = [senPose[0]], [senPose[1]] #the "first waypoint is the initial vehicle pos"
    waypoints = np.zeros(h.config.H) #init waypoints data structure
    ctrlPolicy = [AUV_XY[auvID-1,i] for i in range(3)]+[0.0]*((h.config.H + 1) * 2)
    old_pi_bar = ctrlPolicy
    
    rospy.sleep(1)
    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():
        
        checkMeasNew = measRx[0]
        checkMeasOld = measRxOld[0]
        #check if the a new measurement is received
        if checkMeasNew[0] - checkMeasOld[0] > tol or checkMeasNew[1] - checkMeasOld[1] > tol:
            for i in range(len(measRx)):
                tmp = measRx[i]
                measTable[int(tmp[4])-1].append([tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]])

        if (count1 % (Hz/t_scaler)) == 0:#count seconds for TDMA and acoustic sampling
            clkTdma += 1
            clkSmpl += 1

            if (clkSmpl % TM) == 0:#measure
                for i in range(len(targetsData)): 
                    targetInfo = targetsData[i]
                    if targetInfo[3] != 0:#check if the target actually exist
                        # Perform measurement 
                        [measure, relBearing, measSenPos] = auv.measureBearing(targetInfo[0],
                                                                                targetInfo[1],
                                                                        [senPose[0],senPose[1]],
                                                                        senPose[2])
                        m = [t,measure,measSenPos[0],measSenPos[1],targetInfo[4]]
                        measTx.append(m)
                        measTable[int(targetInfo[4])-1].append(m)
                
            if auvID*Ts == clkTdma:#transmit informations
                f_senPose = [f'{val:.2f}' for val in senPose]
                rospy.loginfo('%s|---- AUV '+str(auvID)+': %s current state %s %s',
                            cyan,auvID,f_senPose,none)
                f_t = f'{t:.2f}'
                rospy.loginfo('%s|---- AUV '+str(auvID)+
                            ': Transmitting measurements at time %s --> Channel Busy%s',
                            cyan,f_t,none)
                
                if len(measTx) > h.config.buffLen:
                    #remove old measurements (max three meas at time otherwise too many bytes)
                    measTx.pop(0)

                measTx = np.array(measTx,dtype=np.float32)
                rows, cols = measTx.shape
                pub[0].publish(Matrix(data=measTx.flatten().tolist(), rows=rows, cols=cols))
                pub[3].publish(np.array(ctrlPolicy,dtype=np.float32))
                measTx = []#empty the buffer of local measures
                if clkTdma == auvNum*Ts:
                    clkTdma = 0

                # Process the measurements and compute target state estimation if some conditions
                for i in range(targetNum):
                    
                    targetInfo = targetsData[i]
                    if targetInfo[3] != 0:#check if the target actually exist
                        if len(measTable[i]) > P_min:  #to be sure there are enough measurements avoiding sing matrix

                            obs[i].processMeasurement(h.orderByTimestamp(measTable[i]))
                            phi,y = obs[i].regressor #update the regressor
                            measTable[i] = []#empty the measurements table
                            
                            # if good conditioning do estimation
                            if h.utils.compute_cost(phi) < thresh:
                                
                                obs[i].propagate_estimation(t) #you can now propagate     
                                cov = h.utils.computeCov(y,phi)#compute a-posteriori cov (vedi paper)
                                confirmedEst = [np.floor(t), targetInfo[4]]#timestamp, label
                                for j in range(4): 
                                    confirmedEst.append(obs[i].state[j,0])
                                for j in range(4):
                                    for k in range(4):
                                        confirmedEst.append(cov[j,k])
                                msgTx.append(confirmedEst)
                    
                # TRIGGER THE OPTIMIZATION IF NEW ESTIMATIONS DONE + SAVE TRACKING DATA ########################    
                if msgTx != []:
                    msgTx = np.array(msgTx,dtype=np.float32)
                    rows, cols = msgTx.shape
                    pub[1].publish(Matrix(data=msgTx.flatten().tolist(), rows=rows, cols=cols))

                    for i in range(len(msgTx)):

                        xi_hat_i = msgTx[i]
                        targetPose = targetsData[i]
                        trackErr[i].append(np.sqrt((targetPose[0] - xi_hat_i[0])**2
                                                    +(targetPose[1] - xi_hat_i[1])**2))
                        f_xi_hat_i = [f"{val:.2f}" for val in xi_hat_i[2:6]]
                        rospy.logout('%s|---- AUV '+str(auvID)+': Target '+str(int(xi_hat_i[1]))
                                    +' state Estimation [m,m/s] --> %s%s',
                                blue,f_xi_hat_i,none)#TODO print the estimate not the msg
                        
                    msgTx = [] #empty the list after sending al the msgs
                    ############################################################################################################
    
        #if the optimization has produced somthing update path, do this control always to avoid unnecessary waitings.
        if sum(ctrlPolicy) != sum(old_pi_bar):
            
            waypoints = ctrlPolicy[7:(len(ctrlPolicy)-1)]
            ax, ay = [senPose[0]], [senPose[1]]#the "first waypoint is the initial vehicle state"
            path, idx_motion, idx, rx, ry, ryaw, surge = h.updatePathRoutine(ax,ay,
                                                            waypoints,senPose,v_n,dt,DT)
            print(path)#TODO, check path routine!!!!!!!!!!! adjust v_n for work with policy
        if path != None: 
            heading.append(ryaw[idx_motion+idx])
            pub[2].publish(np.array([int(auvID),rx[idx_motion+idx],
                                        ry[idx_motion+idx],ryaw[idx_motion+idx]], dtype=np.float32))
            # PUBLISH THE CTRL_CMD
            if len(rx)-1 <= idx_motion+idx:
                idx_motion += 0
            else:
                if auvID != 2:
                    idx_motion += 1  
                else:
                    #SIMULATE AUV2_failure
                    if t > h.config.TIME_DURATION/2 and AUV_failure == True:
                        idx_motion += 0
                    else:
                        idx_motion += 1  
        
        if int(t) == (h.config.TIME_DURATION-1):
            rospy.on_shutdown(lambda: shutdownCllbk(targetNum))
            rospy.signal_shutdown('Simulation time limit reached')

        measRxOld, old_pi_bar = measRx, ctrlPolicy
        t += dt
        count1 += 1 
        if clkTdma >= Tf:
            clkTdma = 0
        
        rate.sleep()

def shutdownCllbk(targetNum):
    global auvID
   
    '''PUT DATA SAVING HERE'''
    for i in range(targetNum):
        np.savetxt(log_path+'/'+str(auvID)+'-trackErr.txt',trackErr[i])


    np.savetxt(log_path+'/'+str(auvID)+'surge_vel.txt',surge_vel)
    np.savetxt(log_path+'/'+str(auvID)+'heading.txt',heading)
    magenta = "\033[0;35m"
    none = "\033[0m"

    rospy.loginfo('%s|---- AUV '+str(auvID)+': Simulation data saved --> Shutting down ...%s',
                    magenta,none)

def callbackSenState(data):
    
    global senPose
    senPose = data.data

def callbackTargetState(data):
    
    global targetsData
    targetsData = np.array(data.data).reshape(data.rows, data.cols)


def callbackCtrlPolicy(data):

    global ctrlPolicy
    ctrlPolicy = data.data
    
def callbackRxMeas(data):
    
    global measRx 
    measRx = np.array(data.data).reshape(data.rows, data.cols)

def listener(auvID):

    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callbackSenState)
    rospy.Subscriber('/'+str(auvID)+'/target_state', Matrix, callbackTargetState)
    rospy.Subscriber('/'+str(auvID)+'/ctrl_policy', numpy_msg(Floats), callbackCtrlPolicy)
    rospy.Subscriber('/'+str(auvID)+'/rx_meas', Matrix, callbackRxMeas)
    
    
def main():

    # ROS INIT   
    namespace = rospy.get_namespace()
    params_path = namespace+'auv'
    # Get AUV ID and number of vehicles.
    global auvID
    auvID = rospy.get_param(params_path+'/auvID')
    auvNum = rospy.get_param(params_path+'/auvNum')
    targetNum = rospy.get_param(params_path+'/targetNum')
    
    # Node Init
    rospy.init_node('auv'+str(auvID)) #TO ADD debug prints --> log_level=rospy.DEBUG

    # Publishers init
    pub = []
    pub_measurement = rospy.Publisher('/'+str(auvID)+'/tx_meas', Matrix, queue_size=100)
    pub_estimation = rospy.Publisher('estimation', Matrix, queue_size=100)
    pub_ctrl_cmd = rospy.Publisher('ctrl_cmd_'+str(auvID), numpy_msg(Floats),queue_size=10)
    pub_ctrl_policy = rospy.Publisher('/'+str(auvID)+'/tx_ctrl_policy',
                                    numpy_msg(Floats), queue_size=100)
    pub.append(pub_measurement)
    pub.append(pub_estimation)  
    pub.append(pub_ctrl_cmd)
    pub.append(pub_ctrl_policy)

    # Initialize sensor and tracker object from costum class
    auv = h.sensor.Sensor(str(auvID),1,0,h.config.SIGMA_MEAS)
    obs = [h.tracker.Tracker(i+1) for i in range(targetNum)]

    # Init Communication protocol parameters (TDMA)
    Tf = h.config.Ts*auvNum
    

    # Start simulation
    listener(auvID)
    run_auv_node(pub,auv,obs,h.config.Ts,Tf,auvNum)
    rospy.on_shutdown(lambda: shutdownCllbk(targetNum))
    rospy.spin()

if __name__ == '__main__':
    main()


