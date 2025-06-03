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
from std_srvs.srv import Trigger

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
trackErr = [[] for _ in range(targetNumSim)]
xi_hat = [[] for _ in range(targetNumSim)]
covariance = [[] for _ in range(targetNumSim)]
# Empty list for ETC statistics
etcEstimation = []
etcGuidance = []

# Weighted distance metric (ETC trigger)
def weighted_distance(seq1, seq2, alpha=0.8):
    w = np.array([alpha**(h) for h in range(len(seq1))])
    d_i = 0.0
    
    for i in range(len(seq1)):
        tmp_seq1 = np.array(seq1[i])
        tmp_seq2 = np.array(seq2[i])
        d_i += w[i]*np.linalg.norm(tmp_seq1 - tmp_seq2)
    
    return np.sum(d_i)

def systemModel(senPose, U, H, dt):

    s = [senPose[0], senPose[1], senPose[2]] # [x,y,theta]
    # Initialize the list to store the predicted states
    s_hat = []
    # Compute new headingRef according to the given heading change

    for i in range(H-1):

        s[2] = s[2]+(U[3+H+i])       
        s[0] = s[0]+np.cos(s[2])*U[3+i]*(dt)
        s[1] = s[1]+np.sin(s[2])*U[3+i]*(dt)
        s_hat.append([s[0],s[1],s[2]])

    return s_hat

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
    global senPose, targetsData, measRx, auvID, ctrlPolicy, etcEstimation, etcGuidance

    # ROS simulation parameters
    t_scaler, TM = h.config.TIME_SCALER, h.config.TM

    Hz = 1/(h.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)

    # Colors for prints
    blue = "\033[1;34m"
    cyan = "\033[0;36m"
    BGreen="\[\033[1;32m\]" 
    BRed = "\033[1;31m"
    none = "\033[0m"

    # Load simulation params from config file
    dt, k_thresh, desRange = h.config.TIME_STEP*t_scaler, h.config.k_phi_thresh, h.config.RANGE_TO_TARGET
    AUV_failure, P_min, H, k_stopCondition = h.config.AUV_failure, h.config.P_min, h.config.H, h.config.k_stopCondition
    DT = h.config.Ts*auvNum
    targetNum = len(obs)

    # Init time variables and counters and lists
    t, count1, clkTdma, clkSmpl, tol = 0,0,0,0,1,
    measTx, msgTx = [], []    
    
    # Initialize ETC routine structures ETC
    etcRoutine = h.etc.EventHandler(targetNum,H,DT)

    # Data structurestargets estimation
    measTable = [[] for _ in range(len(targetsData))]
    measRxOld = [[0,0,0,0,0] for _ in range(targetNum)]
    n_m = h.config.n_m
    # Init bool
    path = None
    missionDone = False

    # ETC on or off
    etcActive = h.config.etcActive
 
    # Start listeners and init waypoints data structure
    ctrlPolicy = [AUV_XY[auvID-1,i] for i in range(3)]+[0.0]*((h.config.H + 1) * 2)
    old_pi_bar = ctrlPolicy
    
    # Wait for start signal from the simulation node
    if wait_for_start_signal():
        rospy.loginfo("Starting agent operations.")

        ## SIMULATION LOOP ############################################################################################################
        while not rospy.is_shutdown():
            
            checkMeasNew = measRx[0]
            checkMeasOld = measRxOld[0]
            #check if the a new measurement is received
            if checkMeasNew[0] - checkMeasOld[0] > tol or checkMeasNew[1] - checkMeasOld[1] > tol:
                if auvID == 3:
                    print('AUV 3 DEBUG: rECEIVED MEASUREMENTS')
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

                    if etcActive == True:
                        if etcRoutine.decisionGuidance == True:
                            etcRoutine.decisionGuidance = False
                            n_m = h.config.n_m + 3
                        else:
                            n_m = h.config.n_m

                    if len(measTx) > n_m:
                        #remove old measurements, they will not be transmitted
                        for i in range(len(measTx)-n_m):
                            measTx.pop(0)

                    measTx = np.array(measTx,dtype=np.float32)
                    rows, cols = measTx.shape

                    pub[0].publish(Matrix(data=measTx.flatten().tolist(), rows=rows, cols=cols))
                    pub[3].publish(np.array(ctrlPolicy,dtype=np.float32))
                    measTx = []#empty the buffer of local measures to transmit
                    if clkTdma == auvNum*Ts:
                        clkTdma = 0

                    # Process the measurements and compute target state estimation if conditions are met
                    for i in range(targetNum):
                        target_info = targetsData[i]

                        # Check if the target exists and has enough measurements
                        if target_info[3] == 0 or len(measTable[i]) <= P_min:
                            pass  # Explicitly do nothing if the conditions are not met
                        else:
                            # Update regressor and clear measurements
                            obs[i].processMeasurement(h.orderByTimestamp(measTable[i]))
                            phi, y = obs[i].regressor
                            measTable[i] = []

                            # Compute conditioning estimation and log it
                            k_phi = h.utils.computeCost(phi)
                            rospy.loginfo('%s|---- AUV %s Conditioning Estimation %s %s', blue, auvID, k_phi, none)

                            if k_phi < k_thresh:
                                # Propagate state and compute covariance
                                obs[i].propagation(t)
                                cov = h.utils.computeCov(y, phi)
                                covariance[i].append(cov)

                                # Prepare estimation message
                                confirmed_est = [np.floor(t), k_phi, target_info[4]]
                                confirmed_est += [obs[i].state[j, 0] for j in range(4)]
                                confirmed_est += [cov[j, k] for j in range(4) for k in range(4)]
                                msgTx.append(confirmed_est)

                                # Run the ETC routine for the consensus algorithm
                                etcRoutine.etcRoutineConsensus(t,i,obs[i].state,cov)
                                if etcRoutine.decisionConsensus:
                                    etcEstimation.append(t)
                                    print('-------------------- ETC: Update Estimation - AUV ID',auvID)
                    
                    # TRIGGER THE OPTIMIZATION IF NEW ESTIMATIONS DONE + SAVE TRACKING DATA ########################    
                    if msgTx != []:
                        msgTx = np.array(msgTx,dtype=np.float32)
                        rows, cols = msgTx.shape
                        
                        for i in range(len(msgTx)):
                            
                            xi_hat_i = msgTx[i]
                            targetPose = targetsData[i]
                            d_s_xi = np.sqrt((xi_hat_i[4]-senPose[1])**2+(xi_hat_i[3]-senPose[0])**2)
                        
                            trackErr[i].append(np.sqrt((targetPose[0] - xi_hat_i[3])**2
                                                        +(targetPose[1] - xi_hat_i[4])**2))
                            xi_hat[i].append(xi_hat_i[3:7])

                            # Log mission status
                            f_xi_hat_i = [f"{val:.2f}" for val in xi_hat_i[3:7]]
                            rospy.logout('%s|---- AUV '+str(auvID)+': Target '+str(int(xi_hat_i[2]))
                                        +' State Estimation [m,m/s] --> %s Range Target %s  %s',
                                    blue,f_xi_hat_i,d_s_xi,none)#TODO print the estimate not the msg

                        # TODO: Now the stop condition is not working for the MTT
                        if d_s_xi <= desRange and k_phi < k_stopCondition:
                            rospy.loginfo('%s|---- AUV '+str(auvID)+' MISSION ACCOMPLISHED')
                            missionDone = True
                        else:
                            rospy.loginfo('%s|---- AUV '+str(auvID)+' OPTIMIZATION STARTING')
                            pub[1].publish(Matrix(data=msgTx.flatten().tolist(), rows=rows, cols=cols))
                            missionDone = False
                        msgTx = [] #empty the list after sending al the msgs
                        ############################################################################################################
        
            # Apply new optimized control if produced
            if sum(ctrlPolicy) != sum(old_pi_bar):
                
                if AUV_failure == True and t > h.config.TIME_DURATION/2 and auvID == 3:
                    rospy.logout('%s|---- AUV '+str(auvID)+': AUV1 Failure%s',BRed,none)
                    ctrlPolicy = [0.0 for _ in range(3+2*(H+1))]
                f_ctrlPolicy = [f"{val:.2f}" for val in ctrlPolicy]
                rospy.logout('%s|---- AUV '+str(auvID)+': Optimization Done!, Output Policy [state (X,Y,Theta), headings (rad), surge (m/s)] --> %s%s',
                BGreen, f_ctrlPolicy, none)
                
                path, idxMotion, idx, rx, ry, ryaw = h.updatePathRoutine(auvID,senPose,
                                                                ctrlPolicy[3:3+H+1],ctrlPolicy[3+H+1:-1],dt,DT)
                # Run the ETC routine for the guidance algorithm
                etcRoutine.etcRoutineGuidance(t,senPose,ctrlPolicy)
                if etcRoutine.decisionGuidance == True:
                    etcGuidance.append(t)
                    print('-------------------- ETC: Update Guidance - AUV ID',auvID)
    

            if path != None and missionDone == False: 
               
                # PUBLISH THE CTRL_CMD           
                pub[2].publish(np.array([int(auvID),rx[idxMotion+idx],
                                            ry[idxMotion+idx],ryaw[idxMotion+idx]], dtype=np.float32))
                
                if idxMotion + idx < len(rx) - 1:
                    if auvID != h.config.failingAUV or not (t > h.config.TIME_DURATION / 2 and AUV_failure):
                        idxMotion += 1
           
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
        np.savetxt(log_path+'/'+str(auvID)+'-x_hat_'+str(targetNum)+'.txt',xi_hat[i])
        # Save the covariance data to a file
    
    np.save(log_path+'/'+str(auvID)+'-cov'+str(targetNum)+'.npy', covariance[i])
    np.savetxt(log_path+'/'+str(auvID)+'etcGuidance.txt',etcGuidance)
    np.savetxt(log_path+'/'+str(auvID)+'etcEstimation.txt',etcEstimation)
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
    
def wait_for_start_signal():
    rospy.loginfo("Waiting for start signal from simulation...")
    rospy.wait_for_service('/start_simulation_service')
    try:
        start_simulation = rospy.ServiceProxy('/start_simulation_service', Trigger)
        response = start_simulation()
        if response.success:
            rospy.loginfo(response.message)
            return True
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
    return False
    
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
    pub_ctrl_cmd = rospy.Publisher('/'+str(auvID)+'/ctrl_cmd', numpy_msg(Floats),queue_size=10)
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


