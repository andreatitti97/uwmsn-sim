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
# Modules for DTW
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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
xi_hat = [[] for _ in range(targetNumSim)]
covariance = [[] for _ in range(targetNumSim)]
# Empty list for ETC statistics
etcEstimation = []
etcGuidance = []


# Compute DTW (Dynamic Time Warping) distance between two 2D control sequences
def compute_dtw(U_k, U_k1):
    """
    Compute the DTW distance between two control sequences U_k and U_k1.
    Each sequence is a 2D array (time steps with surge and sway).
    """
    # Ensure U_k and U_k1 are 2D arrays
    U_k = np.array(U_k).reshape(-1, 2)  # Ensure each control point is 2D
    U_k1 = np.array(U_k1).reshape(-1, 2)

    # Convert the control sequences into a list of tuples for DTW
    U_k_tuples = [tuple(x) for x in U_k]  # Each control point is a 2D tuple
    U_k1_tuples = [tuple(x) for x in U_k1]
    
    distance, _ = fastdtw(U_k_tuples, U_k1_tuples, dist=euclidean)
    return distance

# Decide whether to retransmit based on the DTW distance and adaptive threshold
def retransmit_decision(U_k, U_k1, base_threshold, packet_loss_factor, latency_factor, consensus_error, alpha):
    """
    Decide whether to retransmit based on the discrepancy between control sequences (DTW) 
    and the adaptive threshold considering packet loss, latency, and consensus error.
    """
    # Compute the DTW distance between U_k and U_k1
    dtw_distance = compute_dtw(U_k, U_k1)
    
    # Update the adaptive threshold based on consensus error
    adaptive_threshold = base_threshold #- alpha * consensus_error

    retransmit = dtw_distance >= adaptive_threshold
    return retransmit, adaptive_threshold, dtw_distance




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
    AUV_failure, P_min, H = h.config.AUV_failure, h.config.P_min, h.config.H
    DT = h.config.Ts*auvNum
    targetNum = len(obs)

    # Init time variables and counters and lists
    t, count1, clkTdma, clkSmpl, tol, countETC = 0,0,0,0,1,0
    measTx, msgTx = [], []
    # data structures ETC
    cov_tilde = [np.matrix([[0],[0],[0],[0]]) for _ in range(targetNum)]
    xi_hat_tilde = [[] for _ in range(targetNum)]
    initEtcEstimation = False
    initEtcGuidance = False

    # Data structurestargets estimation
    measTable = [[] for _ in range(len(targetsData))]
    measRxOld = [[0,0,0,0,0] for _ in range(targetNum)]
    # Init bool
    path = None
    missionDone = False
    once = True
    # ETC on or off
    etcActive = h.config.etcActive
    countETC = 0
    decision = True
 
    # Start listeners and init waypoints data structure
    ax, ay = [senPose[0]], [senPose[1]] #the "first waypoint is the initial vehicle pos"
    
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

                    if etcActive == False:
                        for i in range(len(measTx)-1):
                            measTx.pop(0)
                    else:
                        if decision or countETC==H: 
                            for i in range(len(measTx)-1):
                                measTx.pop(0)#remove measurements iff necessary to transmit etc

                    measTx = np.array(measTx,dtype=np.float32)
                    rows, cols = measTx.shape
                    pub[0].publish(Matrix(data=measTx.flatten().tolist(), rows=rows, cols=cols))
                    pub[3].publish(np.array(ctrlPolicy,dtype=np.float32))
                    measTx = []#empty the buffer of local measures
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

                            # Compute cost and log it
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
                                t_est = t

                                # ETC mechanism
                                if not initEtcEstimation:
                                    cov_tilde[i] = cov
                                    xi_hat_tilde[i] = obs[i].state
                                    initEtcEstimation = True
                                else:
                                    delta_t = t - t_est
                                    F = np.array([
                                        [1, 0, delta_t, 0],
                                        [0, 1, 0, delta_t],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]
                                    ])

                                    xi_hat_tilde[i] = F @ xi_hat_tilde[i]
                                    cov_tilde[i] = F @ cov_tilde[i] @ F.T

                                    kld_trace = np.trace(np.linalg.inv(cov) @ cov_tilde[i] - np.eye(4))
                                    kld_logdet = np.log(np.linalg.det(cov) / np.linalg.det(cov_tilde[i]))
                                    kld_norm = np.linalg.norm(xi_hat_tilde[i] - obs[i].state)
                                    KLD = 0.5 * (kld_trace + kld_norm + kld_logdet + 4)

                                    if KLD > 10:
                                        cov_tilde[i] = cov
                                        xi_hat_tilde[i] = obs[i].state
                                        etcEstimation.append(t)
                                        print('---------------------- ETC: Update Estimation')
                                    else:
                                        t_est = t
                    
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


                            f_xi_hat_i = [f"{val:.2f}" for val in xi_hat_i[3:7]]
                            rospy.logout('%s|---- AUV '+str(auvID)+': Target '+str(int(xi_hat_i[2]))
                                        +' State Estimation [m,m/s] --> %s Range Target %s  %s',
                                    blue,f_xi_hat_i,d_s_xi,none)#TODO print the estimate not the msg

                        # TODO: Now the stop condition is not working for the MTT
                        if d_s_xi <= desRange and k_phi < 6.0:
                            rospy.loginfo('%s|---- AUV '+str(auvID)+' MISSION ACCOMPLISHED')
                            missionDone = True
                        else:
                            rospy.loginfo('%s|---- AUV '+str(auvID)+' OPTIMIZATION STARTING')
                            pub[1].publish(Matrix(data=msgTx.flatten().tolist(), rows=rows, cols=cols))
                            missionDone = False
                        msgTx = [] #empty the list after sending al the msgs
                        ############################################################################################################
        
            #if the optimization has produced somthing update path, do this control always to avoid unnecessary waitings.
            if sum(ctrlPolicy) != sum(old_pi_bar):
                
                if AUV_failure == True and t > h.config.TIME_DURATION/2 and auvID == 3:
                    rospy.logout('%s|---- AUV '+str(auvID)+': AUV1 Failure%s',BRed,none)
                    ctrlPolicy = [0.0 for _ in range(3+2*(H+1))]
                f_ctrlPolicy = [f"{val:.2f}" for val in ctrlPolicy]
                rospy.logout('%s|---- AUV '+str(auvID)+': Optimization Done!, Output Policy [state (X,Y,Theta), headings (rad), surge (m/s)] --> %s%s',
                BGreen, f_ctrlPolicy, none)
                
                tmp = []
                for i in range(H):
                    tmp.append([ctrlPolicy[3+i], ctrlPolicy[4+H+i]])
                U_k = [tuple(u) for u in tmp]

                path, idxMotion, idx, rx, ry, ryaw = h.updatePathRoutine(auvID,senPose,
                                                                ctrlPolicy[3:3+H+1],ctrlPolicy[3+H+1:-1],dt,DT)
                if initEtcGuidance == False:
                    tmp = []
                    for i in range(H):
                        tmp.append([ctrlPolicy[3+i], ctrlPolicy[4+H+i]])
                    U_k1 = [tuple(u) for u in tmp]
                    initEtcGuidance = True
                else:

                    decision, thresh, dtw_distance = retransmit_decision(U_k,U_k1,1.0,0.1,0.1,0.1,0.1)
                    if decision or countETC==H:
                        etcGuidance.append(t)
                        if auvID == 1:
                            print('---------------------- ETC: Update Guidance')
                        countETC = 0
                    else:
                        countETC += 1
                    tmp = []
                    for i in range(H):
                        tmp.append([ctrlPolicy[3+i], ctrlPolicy[4+H+i]])
                    U_k1 = [tuple(u) for u in tmp]

            if path != None and missionDone == False: 
               
                # PUBLISH THE CTRL_CMD
                heading.append(ryaw[idxMotion+idx])              
                pub[2].publish(np.array([int(auvID),rx[idxMotion+idx],
                                            ry[idxMotion+idx],ryaw[idxMotion+idx]], dtype=np.float32))
                
                if len(rx)-1 <= idxMotion+idx:
                    idxMotion += 0
                else:
                    if auvID != h.config.failingAUV:
                        idxMotion += 1  
                    else:
                        if t > h.config.TIME_DURATION/2 and AUV_failure == True:
                            idxMotion += 0
                        else:
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

    np.savetxt(log_path+'/'+str(auvID)+'surge_vel.txt',surge_vel)
    np.savetxt(log_path+'/'+str(auvID)+'heading.txt',heading)
    np.savetxt(log_path+'/'+str(auvID)+'etcGuidance.txt',etcGuidance)
    np.savetxt(log_path+'/'+str(auvID)+'etcEstimation.txt',etcEstimation)
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


