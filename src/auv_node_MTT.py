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

# Load the header file as a Python module 
spec = importlib.util.spec_from_file_location("module.header", header_file+'/auv_node_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)
splinePlanner = header.planner

# Init empy lists for saving simulation data
x_hat_1,x_hat_2,x_hat_3,x_hat_4 = [], [], [], []
cov1, cov2, cov3, cov4, err = [], [], [], [], []
heading, surge_vel = [], []

# Init Global Variables for ROS callbacks
AUV_XY = header.config.AUV_XY
targetNum = header.config.targetNum
senPose = [0,0,0] # Sensor state [px,py,yaw]
consMat = []
for i in range(targetNum):
    consMat.append([0,0,0,0]) # --> Target ground truth [px,py,yaw,exist=bool,label]

measRx = [0,0,0,0,0] # --> received measurament [t,meas,ps_x,ps_y,label]
ctrlPolicy = []
for i in range(((len(senPose)+(header.config.H+1)*2))):
    ctrlPolicy.append(0)

def updatePathRoutine(ax,ay,waypoints,s,v_n,dt,DT):

    n_samples = 4
    # Initialized starting position
    a_i = [s[0],s[1]]
    t_i = s[2]

    # Compute new waypoints according to the given heading change
    for i in range(len(waypoints)):
        for j in range(n_samples): #more samples for better curve fitting()
            t_f = t_i+(waypoints[i]/int(DT/(DT/n_samples)))           
            ax.append(np.cos(t_f)*v_n*(DT/n_samples)+a_i[0])
            ay.append(np.sin(t_f)*v_n*(DT/n_samples)+a_i[1])
            a_i = [ax[-1],ay[-1]]
            t_i = t_f

    # Generate new path 
    path = splinePlanner.CubicSpline2D(ax, ay)
    [rx, ry, ryaw, rk, s, surge] = header.utils.calc_spline_course(path,dt)

    tmp = []
    for i in range(len(rx)):
        
        tmp.append(np.sqrt((s[0]-rx[i])**2+(s[1]-ry[i])**2))
        
    idx = tmp.index(min(tmp))
    idx_motion = 0

    return path, idx_motion, idx, rx, ry, ryaw, surge

def run_auv_node(pub,auv,obs,Ts,Tf,auvNum):

    """Simulate the header.sensor platform and the moving target
    Input:              
            pub : list containing the publishers
            auv : list containing sensors state and methods for measurements
            obs : object containing already initialized classes Tracker() (reproduce the local estimations)
            Ts  : slot time of the TDMA protocol
            Tf  : frame time of the TDMA protocol
            auvNum : number ora AUVs
    """
    global senPose, consMat, measRx, auvID, ctrlPolicy

    # ROS simulation parameters
    t_scaler = header.config.TIME_SCALER
    TM = header.config.TM

    Hz = 1/(header.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)

    # Colors for prints
    blue = "\033[1;34m"
    cyan = "\033[0;36m"
    none = "\033[0m"

    # Init time variables and counters and lists
    t, count1, clkTdma, clkSmpl = 0,0,0,0
    measTable, local_measures = [], []
    measRx_old = [0,0,0,0]
    path = None

    # Start listeners and init waypoints data structures
    AUV_failure = header.config.AUV_failure

    ax = [senPose[0]] #the "first waypoint is the initial vehicle state"
    ay = [senPose[1]]
    waypoints = np.zeros(header.config.H) #init waypoints data structure

    old_pi_bar = ctrlPolicy

    # Load simulation params from config file
    dt = header.config.TIME_STEP*t_scaler
    DT = header.config.DT #Optimization Time Window
    thresh = header.config.k_phi_thresh
    thresh = 10000000
    rospy.sleep(1)
    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():
        
        if count1 <= (Hz/t_scaler):#be sure to receive the target and sensor pose at the beginning of the sim
            targetPose = consMat[0]#TODO first target, this part should be avoided 
            d_max = np.sqrt((targetPose[1]-senPose[1])**2+(targetPose[0]-senPose[0])**2)
        
        if (count1 % (Hz/t_scaler)) == 0:#count seconds for TDMA and sampling
            clkTdma += 1
            clkSmpl += 1

            if (clkSmpl % TM) == 0:
                for i in range(len(consMat)): 
                    targetInfo = consMat[i]

                    if targetInfo[3] != 0:#the target actually exist
                        # Perform measurement 
                        [measure_, rel_bearing_, meas_pos] = auv.measureBearing(targetInfo[0],targetInfo[1],
                                                                        [senPose[0],senPose[1]],senPose[2])

                        local_measures.append([t,measure_,meas_pos[0],meas_pos[1],targetInfo[4]])

            if auvID*Ts == clkTdma:
                rospy.loginfo('AUV ID: %s current state %s',auvID,senPose)
                rospy.loginfo('%s|---- AUV '+str(auvID)+': Transmitting measurements at time %s --> Channel Busy%s',
                                cyan,t,none)
                
            
                pub[0].publish(np.array(local_measures,dtype=np.float32))
                pub[3].publish(np.array(ctrlPolicy,dtype=np.float32))
                local_measures = []
                if clkTdma == auvNum*Ts:
                    clkTdma = 0

        if measRx[0] - measRx_old[0] > 1:#check if the measurement is new

            measTable.append(measRx)
            rospy.logdebug('|---- AUV '+str(auvID)+': Measuraments Table [t,y,p_sx,p_sy]--> %s',measTable)

        # Process the measurements and compute target state estimation if some conditions
        if len(measTable) > auvNum: #just to be sure there are enough measurements avoiding sing matrix

            obs.processMeasurement(measTable)
            phi,y = obs.regressor #curr_est at this point is not used!
            measTable = []

        # if good conditioning do estimation 
            if header.utils.compute_cost(phi) >= thresh:
                obs.propagate_estimation(t) #you can now propagate
                curr_est = obs.state
                v_n = header.utils.computePursuitVel(curr_est,senPose,d_max)
                
                rospy.loginfo('OPTIMIZATION ID %s PURSUIT VEL: %s',auvID,v_n)
                cov = header.utils.computeCov(y,phi)
                tmp = []
                for i in range(len(curr_est)):
                    tmp.append(curr_est[i,0])
                tmp.append(v_n)
                #TODO pub[1].publish(np.array(tmp,dtype=np.float32))
                rospy.logout('%s|---- AUV '+str(auvID)+': Target state Estimation [m,m/s] --> %s%s',blue,curr_est,none)
                
                # Compute the tracking error
                
                #TODO: fuse the estimation somewhere err_x = (consMat[0] - curr_est[0,0])
                #TODO err_y = (consMat[1] - curr_est[1,0])
                #TODO e = np.sqrt(err_x**2+err_y**2)
                
                # Save Estimation Data #####################################################################################
                x_hat_1.append(curr_est[0,0])
                x_hat_2.append(curr_est[1,0])
                x_hat_3.append(curr_est[2,0])
                x_hat_4.append(curr_est[3,0])
                #err.append(e)
                # Save Covariance associated 
                cov1.append(cov[0,0])
                cov2.append(cov[1,1])
                cov3.append(cov[2,2])
                cov4.append(cov[3,3])
                ############################################################################################################
  
        #if the optimization has produced somthing update path, do this control always to avoid unnecessary waitings.
        if ctrlPolicy[0] != old_pi_bar[0] and v_n != -10**3:
            
            waypoints = ctrlPolicy[7:(len(ctrlPolicy)-1)]
            ax = [senPose[0]] #the "first waypoint is the initial vehicle state"
            ay = [senPose[1]]

            path, idx_motion, idx, rx, ry, ryaw, surge = updatePathRoutine(ax,ay,
                                                            waypoints,senPose,v_n,dt,DT)
            

        if path != None: 
            heading.append(ryaw[idx_motion+idx])
            surge_vel.append(v_n)
            #if auvID != 2:#for fized node scenario

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
                    if t > header.config.TIME_DURATION/2 and AUV_failure == True:
                        idx_motion += 0
                    else:
                        idx_motion += 1  
        
        if int(t) == (header.config.TIME_DURATION-1):
            rospy.on_shutdown(shutdown_cllbk)
            rospy.signal_shutdown('Simulation time limit reached')

        measRx_old = measRx
        old_pi_bar = ctrlPolicy
        t += dt
        count1 += 1 
        if clkTdma >= Tf:
            clkTdma = 0
        
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

    np.savetxt(log_path+'/'+str(auvID)+'surge_vel',surge_vel)
    np.savetxt(log_path+'/'+str(auvID)+'heading',heading)
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- AUV '+str(auvID)+': Simulation data saved --> Shutting down ...%s',
                    magenta,none)

def callback(data):
    
    global senPose
    senPose = data.data

def callback2(data):
    
    global consMat
    
    consMat = np.array(data.data).reshape(data.rows, data.cols)

def callback3(data):#probably better a service-client paradigm

    global ctrlPolicy
    ctrlPolicy = data.data
    
def callback4(data):
    
    global measRx
    tmp = data.data
    measRx = [tmp[0],tmp[1],tmp[2],tmp[3]]
    
def listener(auvID):

    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callback)
    rospy.Subscriber('/'+str(auvID)+'/target_state', Matrix, callback2)
    rospy.Subscriber('/'+str(auvID)+'/ctrlPolicy', numpy_msg(Floats), callback3)
    rospy.Subscriber('/'+str(auvID)+'/rx_meas', numpy_msg(Floats), callback4)
    
    
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
    pub_ctrl_policy = rospy.Publisher('/'+str(auvID)+'/tx_ctrl_policy', numpy_msg(Floats), queue_size=100)
    pub.append(pub_measurement)
    pub.append(pub_estimation)  
    pub.append(pub_ctrl_cmd)
    pub.append(pub_ctrl_policy)

    # Initialize sensor and tracker object from costum class
    auv = header.sensor.Sensor(str(auvID),1,0,header.config.SIGMA_MEAS)
    obs = header.tracker.Tracker()

    # Init Communication protocol parameters (TDMA)
    Tf = header.config.Ts*auvNum

    # Start simulation
    listener(auvID)
    run_auv_node(pub,auv,obs,header.config.Ts,Tf,auvNum)
    rospy.on_shutdown(shutdown_cllbk)
    rospy.spin()

if __name__ == '__main__':
    main()


