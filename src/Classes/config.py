#!/usr/bin/env python

import numpy as np
import random
import math

# Utils functions
def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_random_points(area, min_distance, max_distance, center_x, center_y):
    # Extracting area dimensions
    area_width, area_height = area
    
    # Generate random points within the specified area
    points = []
    while len(points) < 4:
        # Generate random coordinates within the area
        x = random.uniform(center_x-area_width/2,center_x+area_width/2)
        y = random.uniform(center_y-area_height/2,center_y+area_height/2)
        new_point = (x, y)
        
        # Check if new point is far enough from existing points
        if all(distance_between_points(new_point, existing_point) >= min_distance for existing_point in points):
            # Check if new point is not too far from existing points
            if all(distance_between_points(new_point, existing_point) <= max_distance for existing_point in points):
                points.append(new_point)
    
    return points

def alpha_f(f):
    ''' Compute the term alpha(f) according to Stojanovic'09'''
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

############################################################ SIMULATION SETUP ########################################################
# Simulation parameters
TIME_DURATION = 500 # (s)
TIME_SCALER = 1# in [1 - 10] values near 10 may be source of errors (to fast for ROS stack)
TIME_STEP = 0.01*TIME_SCALER
targetNum = 3 #this is the maximum number of target considered in the simulator
auvNum = 6 #this is the maximum number of auvs considered in the simulator

# Distributed Estimation Algorithm Parameters
TM = 2 #measurements sampling period (s)
P_max = 100 # regressor MAX length 40
P_min = 5 #regressor min length
buffLen = 3 #buffer length for storing received pkts
SIGMA_MEAS = 0.01#0.08#0.1#0.2 # (rad^2) --> 4.5° (as assumed in DAMPS and by cassino)
k_phi_thresh = 30.0 #Thresh sul condizionamento del regressore per aggiornare la stima

# AUVs Team Settings
AUV_MAX_VEL = 1.5 #(m/s) -
AUV_failure = False #auv2 will fail after t = TIME_DURATION/2
AUV2_bridge = True
netTopology = [[] for _ in range(auvNum)]
netTopology[0] = [2] #put the ID of the neigbours of agent 1
netTopology[1] = [1,3] #put the ID of the neigbours of agent 2
netTopology[2] = [2] #put the ID of the neigbours of agent 3

# Randomize initial agents position or chose initial positions
AUV_XY = np.zeros((4,3))
area = (200, 200) # Area dimensions (width, height)
center = (0,0)

random_init = False
min_distance = 35 #Minimum distance between AUVs
max_distance = 500 #Maximum distance between AUVs

if random_init == True:
    random_points = generate_random_points(area, min_distance, max_distance, center[0],center[1])
    
    for i, point in enumerate(random_points):

        AUV_XY[i,0] = point[0]#TODO: SOLVE THE BUG OF HAVING AUV1 IN POS [0,0,0]
        AUV_XY[i,1] = point[1]

else:
    AUV_XY[0,0] = -38
    AUV_XY[0,1] = 200

    AUV_XY[1,0] = +10
    AUV_XY[1,1] = 10

    AUV_XY[2,0] = -150
    AUV_XY[2,1] = -100

# Communication Policy Paramaters
Ts = 4 #TDMA: slot time # time sampling always equal to Ts/2 -- considering pkt=64B and v=480bps
dist = []
for i in range(len(AUV_XY)-1):
    tmp1 = (AUV_XY[i,0],AUV_XY[i,1])
    tmp2 = (AUV_XY[i+1,0],AUV_XY[i+1,1])
    dist.append(distance_between_points(tmp1,tmp2))

avg_d = sum(dist)/(len(dist))
d = avg_d
gamma = avg_d*3 #Sigmoid parameter for packet loss --> depends on the distance (tune only alpha)
PDR = 90

# Acoustic Model Parameters
SL = 200 #db
NL = 20 #db
DI = 0 #directivity index a-dimensional
c = 1500 #sound wave speed
f = 10 #kHx ( frequency of the modem)

for i in range(len(dist)):
    acoustic_loss = alpha_f(f) #f is in kHz
    TL = 20*np.log(dist[i]) + (dist[i]*acoustic_loss*1e-3)
DThresh = 20*np.log(min_distance) + (min_distance*acoustic_loss*1e-3)#dB (transmission loss that if happens is "ideal")
TL_worse = 20*np.log(max_distance) + (max_distance*acoustic_loss*1e-3)

# Optimization Parameters --- alpha = 0.15, gamma = 1.0 (almost fixed formation)
alpha_w = 0.75 #fixed form 0.45#0.15
gamma_w = 0.1#0.3 #fixed form 0.85#0.25#1.0
DThresh = 0 #dB (minimum connectivity requirement)
RANGE_TO_TARGET = 50 

u_max = 25*math.pi/180
delta_u = 5*math.pi/180
MAX = 60*math.pi/180
MIN = 10*math.pi/180
U = 7 # number of control choices (should be an ODD number)
H = 3 # planning horizon

ctrl_cmd = []
u_i = u_max/((U-1)/2)
for i in range(U):
    if i < np.floor(U/2):
        ctrl_cmd.append(-(u_max-i*u_i))
    elif i == np.ceil(U/2):
        ctrl_cmd.append(0.0)
    if i > U/2:
        ctrl_cmd.append((i-((U-1)/2))*u_i)

######## CHOOSE TARGET DYNAMIC ###################################################################################################
# CHOOSE Target parameter: start, goal, min max vels
# PARTE SEMPRE DA UNA DISTANZA COMPRESA TRA I 3.5 E 5 KM con velocità da 4 a 8 m/s
TARGET_INIT = [400,-200, math.pi/2, 3.0, 0.0, 0.0, 0.0] #SIMPLE CASE LOWE DISTANCE!!!!!
#TARGET_INIT = [800, 100, 140*math.pi/180, 3.0, 0.0, 0.0, 0.0]
TARGET_INIT = [+2000,-2500, math.pi, 2.5, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 1
#TARGET_INIT = [4000, 200, 140*math.pi/180, 8.0, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 2
#TARGET_INIT = [-5000,-5000, math.pi/4, 3.0, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 3
#TARGET_INIT = [-500,+3500, -math.pi/10, 2.5, 0.0, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 4
#TARGET_INIT = [3000,-1500, math.pi/2, 9.0, 0.0, 0.5, 0.0]#[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 5
#TARGET_INIT = [-2000, +2000, 140*math.pi/180, 8.0, 0.002, 0.0, 0.0] #[x(m),y(m),theta(rad),v0(m/s),omega0,v_dot0,omega_dot0] - DINAMICA 6
#TARGET_INIT = [-2000, -1800, math.pi/2, 6.0, -0.001, 0.0, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 7
#TARGET_INIT = [-1500, 2000, math.pi/8, 5.0, 0.0, 0.8, 0.0] #[x(m),y(m),theta(rad),linear vel(m/s)] - DINAMICA 8
# PAPER CONTROLO
#TARGET_INIT = [-100,-30, math.pi/2, 0.0, 0.0, 0.0, 0.0] #static
#TARGET_INIT = [-70,+15, math.pi-math.pi/8, 0.8, 0.0, 0.0, 0.0] #ideal moving

TARGET_INIT = [-300,-50, math.pi+math.pi/2-math.pi/6, 0.5, 0.0, 0.0, 0.0] #realistic moving 1
#TARGET_INIT = [-150,+300, math.pi, 0.4, 0.0, 0.0, 0.0] #realistic moving 2
#TARGET_INIT = [-200,+10, math.pi, 0.4, 0.0, 0.0, 0.0] #ideal moving 2
# PAPER JOURNAL
#TARGET_INIT = [-150,0, math.pi+math.pi/2-math.pi/6, 0.5, 0.0, 0.0, 0.0] # validation 1
TARGET_INIT = [-450,105, np.pi/2+np.pi/8, 0.0, 0.0, 0.0, 0.0] # validation 2

alpha_0, omega_0,alpha_dot_0,omega_dot_0 = TARGET_INIT[3],TARGET_INIT[4],TARGET_INIT[5],TARGET_INIT[6]

MAX_TARGET_VEL = 3 #(m/s) (only if target no costant vels)
MIN_TARGET_VEL = 3 #(m/s)
sin_pattern = False

for i in range(len(AUV_XY)):
    AUV_XY[i,2] = math.atan2(TARGET_INIT[1]-AUV_XY[i,1],TARGET_INIT[0]-AUV_XY[i,0])

###################################################################################################################################