#!/usr/bin/env python
from math import pi

############################################################ SIMULATION SETUP ########################################################
# Simulation parameters
TIME_DURATION = 250 # (s)
TIME_STEP = 0.01
TIME_SCALER = 1# in [1 - 10] values near 10 may be source of errors (to fast for ROS stack)
c = 1500 #sound wave speed
OPTIMIZATION_ON = False

# Estimation Parameters
TP = 30 # regressor MAX length 40
buffLen = 10 #buffer length for storing received meas
SIGMA_MEAS = 0.08# (rad^2) --> 4.5° (as assumed in DAMPS and by cassino)

# AUVs Team Settings
d = 20 #vehicle distance
AUV_MAX_VEL = 1.0 #(m/s) - max vel (if CPF active v_coop should be considered)
RANGE_TO_TARGET = 10

# Communication Paramaters
Ts = 2 #TDMA: slot time
alpha = -0.003 #Sigmoid parameters for packet loss, if alpha << gamma --> more packet loss
gamma = d*2 #Sigmoid parameter for packet loss --> depends on the distance (tune only alpha)

DT = 10 #d*auvNum/10 sort of
k_phi_thresh = 1 #Thresh sul condizionamento del regressore per aggiornare la sitma

# Optimization Parameters
DELTA = 10 #to start the BnB algorithm
u_max = 20*pi/180
delta_u = 10*pi/180
MAX = 60*pi/180
MIN = 5*pi/180
U = 3 #number of control choices
H = 3 # planning horizon

ctrl_cmd = [-u_max,-u_max*2/(U/2),-u_max/(U/2),0,
                u_max/(U/2), u_max*2/(U/2), u_max] #set of control actions
#ctrl_cmd = [-u_max, -u_max*8/(U) ,-u_max*6/(U) ,-u_max*4/(U),-u_max*2/(U),0,
                #u_max*2/(U), u_max*4/(U), u_max*8/(U) , u_max*6/(U) ,u_max] #set of control actions
ctrl_cmd = [-u_max,0,u_max]
##################################################################################################################################


######## CHOOSE TARGET DYNAMIC ###################################################################################################
# CHOOSE Target parameter: start, goal, min max vels
# PARTE SEMPRE DA UNA DISTANZA COMPRESA TRA I 3.5 E 5 KM con velocità da 4 a 8 m/s
TARGET_INIT = [400,-200, pi/2, 3.0, 0.0, 0.0, 0.0] #SIMPLE CASE LOWE DISTANCE!!!!!
#TARGET_INIT = [800, 100, 140*pi/180, 3.0, 0.0, 0.0, 0.0]
TARGET_INIT = [+2000,-2500, pi, 2.5, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 1
#TARGET_INIT = [4000, 200, 140*pi/180, 8.0, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 2
#TARGET_INIT = [-5000,-5000, pi/4, 3.0, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 3
#TARGET_INIT = [-500,+3500, -pi/10, 2.5, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 4
#TARGET_INIT = [3000,-1500, pi/2, 9.0, 0.0, 0.5, 0.0]#[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 5
#TARGET_INIT = [-2000, +2000, 140*pi/180, 8.0, 0.002, 0.0, 0.0] #[x(m),y(m),theta(rad),v0(m/s),omega0,v_dot0,omega_dot0] - DINAMICA 6
#TARGET_INIT = [-2000, -1800, pi/2, 6.0, -0.001, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 7
#TARGET_INIT = [-1500, 2000, pi/8, 5.0, 0.0, 0.8, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 8
TARGET_INIT = [-100,-15, pi/2, 0.0, 0.0, 0.0, 0.0] 
TARGET_INIT = [-70,+15, pi-pi/8, 0.0, 0.0, 0.0, 0.0]

alpha_0, omega_0,alpha_dot_0,omega_dot_0 = TARGET_INIT[3],TARGET_INIT[4],TARGET_INIT[5],TARGET_INIT[6]
MAX_TARGET_VEL = 3 #(m/s) (only if target no costant vels)
MIN_TARGET_VEL = 3 #(m/s)
###################################################################################################################################

# DO NOT EDIT ##############################################################

# Team parameter: number of agents, baselines_XY, inital position, type of formation
PLATFORM_INIT_POSE = [0, 0, 0] #[x,y,theta]
geometry = 'line2' #only for formation control
if geometry == 'column' or geometry == 'column2':
    PLATFORM_INIT_POSE = [d, 0, 0]
# Cooperative Path Following Params
K_att = 0.8 # Attractive Gain
K_rep = 0.0 # Repulsive gain
d_rep = d # Distance threshold for repulsion


