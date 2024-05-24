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

############################################################ SIMULATION SETUP ########################################################
# Simulation parameters
TIME_DURATION = 500 # (s)
TIME_STEP = 0.01
TIME_SCALER = 1# in [1 - 10] values near 10 may be source of errors (to fast for ROS stack)
c = 1500 #sound wave speed
OPTIMIZATION_ON = False

# Estimation Parameters
TP = 30 # regressor MAX length 40
buffLen = 10 #buffer length for storing received meas
SIGMA_MEAS = 0.1# #0.08# (rad^2) --> 4.5° (as assumed in DAMPS and by cassino)

# AUVs Team Settings
AUV_MAX_VEL = 2.5#2.0#0.6  #(m/s) -
RANGE_TO_TARGET = 100 #(far mission), 10 near mission

# Optimization Parameters

u_max = 35*math.pi/180
delta_u = 10*math.pi/180
MAX = 60*math.pi/180
MIN = 10*math.pi/180
U = 7 #number of control choices
H = 4 # planning horizon

if U == 7:
    ctrl_cmd = [-u_max,-u_max*2/((U-1)/2),-u_max/((U-1)/2),0,
                u_max/((U-1)/2), u_max*2/((U-1)/2), u_max] #set of control actions
elif U == 5:
    ctrl_cmd = [-u_max,-u_max/2,0,u_max/2,u_max] #set of control actions
else:
    ctrl_cmd = [-u_max,0,u_max]

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

#TARGET_INIT = [-100,-30, math.pi/2, 0.0, 0.0, 0.0, 0.0] #static
#TARGET_INIT = [-70,+15, math.pi-math.pi/8, 0.8, 0.0, 0.0, 0.0] #ideal moving
TARGET_INIT = [-150,0, math.pi+math.pi/2-math.pi/6, 0.3, 0.0, 0.0, 0.0] #realistic moving 1
#TARGET_INIT = [-150,+300, math.pi, 0.4, 0.0, 0.0, 0.0] #realistic moving 2
#TARGET_INIT = [-200,+10, math.pi, 0.4, 0.0, 0.0, 0.0] #ideal moving 2

alpha_0, omega_0,alpha_dot_0,omega_dot_0 = TARGET_INIT[3],TARGET_INIT[4],TARGET_INIT[5],TARGET_INIT[6]
MAX_TARGET_VEL = 3 #(m/s) (only if target no costant vels)
MIN_TARGET_VEL = 3 #(m/s)
###################################################################################################################################


AUV_XY = np.zeros((4,3))

# Example usage
area = (200, 200) #(500,500) # Area dimensions (width, height)
center = (0,0)

min_distance = 50# Minimum distance between AUVs
max_distance = 250# Maximum distance between AUVs

random_points = generate_random_points(area, min_distance, max_distance, center[0],center[1])
dist = []
for i, point in enumerate(random_points):

    AUV_XY[i,0] = point[0]#TODO: SOLVE THE BUG OF HAVING AUV1 IN POS [0,0,0]
    AUV_XY[i,1] = point[1]

    AUV_XY[i,2] = math.atan2(TARGET_INIT[1]-AUV_XY[i,1],TARGET_INIT[0]-AUV_XY[i,0])
    tmp1 = (AUV_XY[i,0],AUV_XY[i,1])
    tmp2 = (TARGET_INIT[0],TARGET_INIT[1])
    dist.append(distance_between_points(tmp1,tmp2))

AUV_XY[0,0] = 5
AUV_XY[0,1] = 75

AUV_XY[1,0] = +75
AUV_XY[1,1] = 10

AUV_XY[2,0] = 5
AUV_XY[2,1] = -75

dist = []

for i in range(3):
    AUV_XY[i,2] = math.atan2(TARGET_INIT[1]-AUV_XY[i,1],TARGET_INIT[0]-AUV_XY[i,0])
    tmp1 = (AUV_XY[i,0],AUV_XY[i,1])
    tmp2 = (TARGET_INIT[0],TARGET_INIT[1])
    dist.append(distance_between_points(tmp1,tmp2))

avg_d = sum(dist)/len(dist)-10
d = avg_d

# Communication Paramaters
Ts = 4 #TDMA: slot time # time sampling always equal to Ts/2
n = 3 #auv num
DT = Ts*n*2

alpha = -0.1 #0.01 #Sigmoid parameters for packet loss, if alpha << gamma --> more packet loss
gamma = avg_d*3 #Sigmoid parameter for packet loss --> depends on the distance (tune only alpha)

# IN REALTÀ PERME CONVIENE METTERE IL CONDIZIONAMENTO INIZIALE COME THRESH
k_phi_thresh = 1 #Thresh sul condizionamento del regressore per aggiornare la sitma