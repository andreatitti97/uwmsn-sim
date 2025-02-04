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
from uwmsn_msgs.msg import Matrix
from std_srvs.srv import Trigger, TriggerResponse

# Environment: Define the relevant paths
'''pathlib: output is an object path (sum a string using '/')
os.path.dirname: output is a string with the path (sum strings using defaul python cmd '+')'''

pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
log_path = os.path.dirname(pkg_directory)+'/logs'
class_path = pkg_directory+'/src/Classes'

# Load the h file as a Python module 
header_file = pathlib.Path(__file__).parent.resolve()
header_file = os.path.dirname(header_file)
header_file = header_file+'/include'+'/uwmsn-sim'
spec = importlib.util.spec_from_file_location("module.header", header_file+'/main-kinematic_h.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

#Load global variables for callback
targetNumSim, auvNumSim, simTime = h.config.targetNum, h.config.auvNum, h.config.TIME_DURATION
Ts, alpha_w, gamma_w, NL = h.config.Ts, h.config.alpha_w, h.config.gamma_w, h.config.NL
H, desRange, ctrl_set = h.config.H, h.config.RANGE_TO_TARGET, h.config.ctrl_cmd
# Init data structures for callback
paths = [[None, None, None] for _ in range(len(h.config.AUV_XY))]
s_traj_x, s_traj_y = [[] for _ in range(auvNumSim)], [[] for _ in range(auvNumSim)]
t_traj_x, t_traj_y = [[] for _ in range(auvNumSim)], [[] for _ in range(auvNumSim)]
global t

def run_simulation(target_list, auvNum, pub_s_state, pub_t_state, pub_init_opt):

    """Simulate the sensor platform and the moving target
    Input:  
            target_list : list containing target initialized structs
            auvNum : list containing sensors state and methods for measurements
            pub_s_state : list containing the publishers for the vehicles states
            pub_t_state : list containing the publishers for the target state

    """
    global count1, paths, t

    # ROS simulation parameters
    t_scaler = h.config.TIME_SCALER
    Hz = 1/(h.config.TIME_STEP) 
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    dt = h.config.TIME_STEP*t_scaler
    targetNum = len(target_list)
     
    # Init AUVs position and orientation
    auvs_xy = h.config.AUV_XY
    initMsg, sensors, measTable = [], [], []

    for i in range(auvNum):

        initMsg.append(auvs_xy[i,0])
        initMsg.append(auvs_xy[i,1])
        initMsg.append(auvs_xy[i,2])
        sensors.append(h.sensor.Sensor(str(i),1,0,0.000))
    
    for i in range(auvNum):
        target = target_list[0]#first target as reference.
        target.exist = True # Spawn the first target at the beginning of the simulation
        [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.pose.x,target.pose.y,
                                                                       [auvs_xy[i,0],auvs_xy[i,1]],
                                                                       auvs_xy[i,2])
        arr = [measure_,meas_pos[0],meas_pos[1]]
        measTable.append(arr)

    estimator = h.estimator_module.Estimation()
    estimator.computeState(measTable)

    for i in range(auvNum):
        rospy.loginfo('|---- KINEMATIC SIMULATION: Initial AUV'+str(i+1)+' pose (m) --> %s',auvs_xy[i,:])
    for i in range(targetNum):
        target = target_list[i]
        rospy.loginfo('|---- KINEMATIC SIMULATION: Initial Target(s) pose (m) --> %s',
                        [target.pose.x,target.pose.y,target.pose.theta])
        


    rospy.sleep(1)

    ## SIMULATION LOOP ############################################################################################################
    while not rospy.is_shutdown():
        if count1 < Hz: #to be sure that the initalization setup is shared among nodes
            pub_init_opt[0].publish(np.array(initMsg,dtype=np.float32)) #ONE TIME PUBLISHER for init state of S
            tmp = []
            
        for i in range(auvNum):
            # Publish agents info
            pub_s_state[i].publish(np.array([auvs_xy[i,0],auvs_xy[i,1],auvs_xy[i,2]],
                                            dtype=np.float32))
            # Publish targets info
            consMat = []
            for j in range(targetNum):
                target = target_list[j]
                consMat.append([target.pose.x,target.pose.y,target.pose.theta,
                                target.exist,target.label])

            consMat = np.array(consMat,dtype=np.float32)
            rows, cols = consMat.shape
            pub_t_state[i].publish(Matrix(data=consMat.flatten().tolist(), rows=rows, cols=cols))

        # Move Agents
        for i in range(auvNum):
            tmp = paths[i]
            if tmp[0] != None:
                
                #auvs_xy[i,0], auvs_xy[i,1], auvs_xy[i,2]  = tmp[0], tmp[1], tmp[2]
                auvs_xy[i,2] = auvs_xy[i,2] + tmp[2]
                auvs_xy[i,0] = auvs_xy[i,0] + np.cos(tmp[0])*tmp[1]*dt
                auvs_xy[i,1] = auvs_xy[i,1] + np.sin(tmp[0])*tmp[1]*dt #

        # Move Targets
        for i in range(targetNum):
            target = target_list[i]
            if target.exist == True:
                target.move_target(dt)
                    
        #################################################################################################################
        ##################### SAVE THE POSITIONS OF TEAM REFERENCE/AGENTS/TARGET/ STATE FOR PLOT ########################
        
        for i in range(auvNum):
            s_traj_x[i].append(auvs_xy[i,0])
            s_traj_y[i].append(auvs_xy[i,1])

        for i in range(targetNum):
            target = target_list[i]
            t_traj_x[i].append(target.pose.x)
            t_traj_y[i].append(target.pose.y)         
        
        ##################################################################################################################
        #  Stop simulation and save data to .txt files ###################################################################
        if int(t) == (h.config.TIME_DURATION-1):
            rospy.on_shutdown(lambda: shutdown_cllbk(auvNum,targetNum,t))
            rospy.signal_shutdown('Simulation time limit reached')
      
        if count1 % Hz == 0:
            '''ADD DEBUG PRINTS HERE'''
            f_t = f'{t:.2f}'
            rospy.loginfo('|---- KINEMATIC SIMULATION: Elapsed time (s) --> %s',f_t)
            for i in range(targetNum):
                target = target_list[i]
                if target.exist == True:
                    rospy.loginfo('|---- KINEMATIC SIMULATION: Target '+str(i+1)+' groud truth (m) --> %s, exist=%s',
                                [target.pose.x,target.pose.y], target.exist)
                
        ''' ADD SPAWNING TARGETS HERE'''
        if t > 0:
            if targetNum > 1:
               target_list[1].exist = True
        if t > 0:
            if targetNum > 2:
                target_list[2].exist = True

        t += dt
        count1 += 1  
        rate.sleep()

def shutdown_cllbk(auvNum,targetNum,t):

    '''PUT DATA SAVING HERE'''
    for i in range(targetNum):  
        np.savetxt(log_path+'/target_x_traj'+str(i+1)+'.txt',t_traj_x[i])
        np.savetxt(log_path+'/target_y_traj'+str(i+1)+'.txt',t_traj_y[i])
        
    for i in range(auvNum):

        np.savetxt(log_path+'/auv_x_traj'+str(i+1)+'.txt',s_traj_x[i])
        np.savetxt(log_path+'/auv_y_traj'+str(i+1)+'.txt',s_traj_y[i])

    # Save a logfile with simulation settings
    sim_info = [Ts, NL, desRange, alpha_w, gamma_w]
    sim_data = [auvNum, targetNum, t, len(t_traj_x[0])]
    np.savetxt(log_path+'/sim_info.txt',sim_info)
    np.savetxt(log_path+'/sim_data.txt',sim_data)
    np.savetxt(log_path+'/ctrl_set.txt',ctrl_set)

    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('|---- %sKINEMATIC SIMULATION: Simulation data saved --> Shutting down ...%s',
                  magenta,none)

# Generic callback for updating the path of a specific AUV
def callback(data, auvIndex):
    global paths
    tmp = data.data
    paths[auvIndex] = [tmp[1], tmp[2], tmp[3]]

def listener(auvNum):
    for i in range(auvNum):
        # Create a unique callback function for each subscriber
        rospy.Subscriber('/'+str(i+1)+'/ctrl_cmd',
                        numpy_msg(Floats), lambda data, i=i: callback(data, i))
    
def handle_start_request(req):
    rospy.loginfo("Start signal sent to agent")
    return TriggerResponse(success=True, message="Simulation started")

def main():

    global t
    # ROS INIT   
    # Get AUV ID and number of vehicles.
    auvNum = rospy.get_param('/kinematic_sim/auvNum')
    targetNum = rospy.get_param('/kinematic_sim/targetNum')
    # Node Init
    rospy.init_node('kinematic_sim') #log_level=rospy.DEBUG
    
    # Initialize publishers
    pub_t_state, pub_s_state ,pub_init_opt= [], [], []
    pub_init_opt.append(rospy.Publisher('/init_opt1', numpy_msg(Floats),
                                         queue_size=100))
    for i in range(auvNum):
        pub_s_state.append(rospy.Publisher('/'+str(i+1)+'/vehicle_state_'+str(i+1),
                                        numpy_msg(Floats), queue_size=100))
        pub_t_state.append(rospy.Publisher('/'+str(i+1)+'/target_state', Matrix,
                                queue_size=100))
        
    # Start service
    start_service = rospy.Service('/start_simulation_service', Trigger, handle_start_request)
       
    # One time publisher or initialize the optmization node with all AUVs info
    
    # Initialization object h.target
    target_list = []
    for i in range(targetNum):
        target_list.append(h.target.Target(i+1,False))


    # Start listeners and run simulation
    listener(auvNum)

    

    run_simulation(target_list, auvNum, pub_s_state, pub_t_state, pub_init_opt)  
    rospy.on_shutdown(lambda: shutdown_cllbk(auvNum,targetNum,t))
    rospy.spin()

if __name__ == '__main__':
    main()


