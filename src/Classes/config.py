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
TIME_DURATION = 3000 # (s)
TIME_SCALER = 1# in [1 - 10] values near 10 may be source of errors (to fast for ROS stack)
TIME_STEP = 0.01*TIME_SCALER
targetNum = 3 #this is the maximum number of target considered in the simulator
auvNum = 6 #this is the maximum number of auvs considered in the simulator

# Distributed Estimation Algorithm Parameters
TM = 2 #measurements sampling period (s), lower than this impossible due to AVS processing!
P_max = 50 # regressor MAX length 40
P_min = 20 #regressor min length

SIGMA_MEAS = 0.04# #(rad^2) --> 4.5° (as assumed in DAMPS and by cassino)| 6.0° super harsh
k_phi_thresh = 80.0# #Thresh sul condizionamento del regressore per aggiornare la stima
k_stopCondition = 4.0# #Thresh sul condizionamento del regressore per fermare l'algoritmo
n_m = 3 #numero di misure trasmesse

# AUVs Team Settings
AUV_MAX_VEL = 5.0 #(m/s)
AUV_failure = False #auv1 will fail after t = TIME_DURATION/2
failingAUV = 3 #ID AUV that will fail 
etcActive = True
alphaETC = 0.05
netTopology = [[] for _ in range(auvNum)]
netTopology[0] = [2] #put the ID of the neigbours of agent 1
netTopology[1] = [1,3] #put the ID of the neigbours of agent 2
netTopology[2] = [2] #put the ID of the neigbours of agent 3

# Randomize initial agents position or chose initial positions
active_auv = 3#choose many AUV you plan to use
AUV_XY = np.zeros((active_auv,3))
area = (200, 200) # Area dimensions (width, height)
center = (0,0)

random_init = False
min_distance = 50 #Minimum distance between AUVs
max_distance = 2500 #Maximum distance between AUVs

if random_init == True:
    random_points = generate_random_points(area, min_distance, max_distance, center[0],center[1])
    
    for i, point in enumerate(random_points):

        AUV_XY[i,0] = point[0]#TODO: SOLVE THE BUG OF HAVING AUV1 IN POS [0,0,0]
        AUV_XY[i,1] = point[1]

else:

    # FAR INITIAL POSITION
    AUV_XY[0,0] = 400
    AUV_XY[0,1] = 400

    AUV_XY[1,0] = 400   
    AUV_XY[1,1] = 10

    AUV_XY[2,0] = 10
    AUV_XY[2,1] = 10

  # FAR INITIAL POSITION alternative
    AUV_XY[0,0] = 430
    AUV_XY[0,1] = 200

    AUV_XY[1,0] = 400   
    AUV_XY[1,1] = 10

    AUV_XY[2,0] = 10
    AUV_XY[2,1] = 10

         # BASE INITIAL POSITION
    AUV_XY[0,0] = -38
    AUV_XY[0,1] = 220

    AUV_XY[1,0] = 10   
    AUV_XY[1,1] = -10 

    AUV_XY[2,0] = -150
    AUV_XY[2,1] = -100 

    # FAR INITIAL POS
    AUV_XY[0,0] = -10
    AUV_XY[0,1] = 10

    AUV_XY[1,0] = -10   
    AUV_XY[1,1] = 350

    AUV_XY[2,0] = -450
    AUV_XY[2,1] = 510

    # VERY FAR INITIAL POS
    AUV_XY[0,0] = -4000
    AUV_XY[0,1] = 1000

    AUV_XY[1,0] = -2000 
    AUV_XY[1,1] = 1000

    AUV_XY[2,0] = 0
    AUV_XY[2,1] = 1000


# Communication Policy Paramaters

dist = []
for i in range(len(AUV_XY)-1):
    
    tmp1 = (AUV_XY[i,0],AUV_XY[i,1])
    tmp2 = (AUV_XY[i+1,0],AUV_XY[i+1,1])
    dist.append(distance_between_points(tmp1,tmp2))

avg_d = sum(dist)/(len(dist))
d = avg_d
gamma = avg_d*3 #Sigmoid parameter for packet loss --> depends on the distance (tune only alpha)

# Acoustic Model Parameters
SL = 186 #we worked with modem at 182 - 168 db
NL = 30 #db
DI = 0 #directivity index a-dimensional
c = 1500 #sound wave speed
PDR = 90 #Packet Delivery Ratio

if max_distance == 2500:
    f = 10 #kHx ( frequency of the modem) long range 2-10 kHz
elif max_distance == 5000:
    f = 6 #kHx ( frequency of the modem) long range 2-10 kHz
elif max_distance == 10000:
    f = 2 #kHx ( frequency of the modem) long range 2-10 kHz

pktSize = 1024+128*n_m #(fixed_pkt_size + 128*n_m)
B = f*1000/20 #bps (bandwidth)
Ts = TM + int(np.floor(pktSize/B)) #TDMA: slot time # time sampling always equal to Ts/2 -- considering pkt=64B and v=480bps
print('TIME SLOT',Ts)

# Acoustic Loss Parameters
acoustic_loss = alpha_f(f) #f is in kHz
TL_min = 20*np.log10(min_distance) + (min_distance*acoustic_loss*1e-3)#dB (transmission loss that if happens is "ideal")
TL_max = 20*np.log10(max_distance) + (max_distance*acoustic_loss*1e-3)

SNR_ub = SL -TL_min - NL - DI
SNR_lb = SL -TL_max - NL - DI
SNR_minimal = 90 #dB TODO validate this value

if SNR_ub < SNR_minimal: 
    print('INCREASE SOURCE LEVEL or TOO MUCH NOISE')
'''print('SNR expected initial', SNR_lb)
print('(SNR lower bound):', SNR_lb)
print('(SNR upper bound):', SNR_ub)'''
for i in range(len(dist)):
    TL = 20*np.log10(dist[i]) + (dist[i]*acoustic_loss*1e-3)
    SNR = SL -TL - NL - DI
    #print('INITIL DIStanceS',dist[i])
    if SNR < SNR_lb or SNR > SNR_ub:

        print('ACOUSTIC PARAMETERS NOT GOOD (SNR, SNR_max, SNR_min)',SNR, SNR_lb, SNR_ub)
    '''else:
        print('SNR '+str(i)+'-'+str(i+1),SL -TL - NL - DI)'''

# Optimization Parameters --- alpha = 0.15, gamma = 1.0 (almost fixed formation)
alpha_w = 0.6 #fixed form 0.45#0.15
gamma_w = 0.1 #fixed form 0.85#0.25#1.0
RANGE_TO_TARGET = min_distance*2 #

U = 5  # number of control choices (should be an ODD number)
u_max = 45*math.pi/180
delta_u = 5*math.pi/180
MAX = 60*math.pi/180
MIN = 10*math.pi/180
H = 3 # planning horizon

# Generate control commands using list comprehension
u_i = u_max / ((U - 1) / 2)
ctrl_cmd = [-u_max + i * u_i for i in range(U)]

######## CHOOSE TARGET DYNAMIC ###################################################################################################
# CHOOSE Target parameter: start, goal, min max vels
# PARTE SEMPRE DA UNA DISTANZA COMPRESA TRA I 3.5 E 5 KM con velocità da 4 a 8 m/s

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


#TARGET_INIT = [-150,+300, math.pi, 0.4, 0.0, 0.0, 0.0] #realistic moving 2
#TARGET_INIT = [-200,+10, math.pi, 0.4, 0.0, 0.0, 0.0] #ideal moving 2
# PAPER JOURNAL
#TARGET_INIT = [-150,0, math.pi+math.pi/2-math.pi/6, 0.5, 0.0, 0.0, 0.0] 
TARGET_INIT = [-300,-50, math.pi+math.pi/2-math.pi/6, 0.3, 0.0, 0.0, 0.0] 
TARGET_INIT = [-250,105, np.pi-np.pi/6, 0.2, 0.0, 0.0, 0.0]
TARGET_INIT = [-250,350, np.pi-np.pi/3, 0.2, 0.0, 0.0, 0.0]
TARGET_INIT = [-350,250, np.pi, 0.2, 0.0, 0.0, 0.0]
TARGET_INIT = [-300,150, np.pi-np.pi/3, 0.2, 0.0, 0.0, 0.0] 
TARGET_INIT = [400,0, np.pi/2, 0.35, 0.0, 0.0, 0.0] 
TARGET_INIT = [400,0, np.pi, 0.2, 0.0, 0.0, 0.0] # SCENARIO 2-3
TARGET_INIT = [-100,-350, -np.pi/2, -0.15, 0.0, 0.0, 0.0] # SCENARIO 1 bonus
TARGET_INIT = [250,350, np.pi, -0.35, 0.0, 0.0, 0.0] # SCENARIO 5
TARGET_INIT = [-300,25, np.pi/3, 0.35, 0.0, 0.0, 0.0] #scenario 4
TARGET_INIT = [0,-350, -np.pi/6, -0.15, 0.0, 0.0, 0.0] # SCENARIO 1
TARGET_INIT = [400,0, np.pi, 0.2, 0.0, 0.0, 0.0] # SCENARIO 2-3
TARGET_INIT = [0,-350, 0, 0.2, 0.0, 0.0, 0.0] # SCENARIO 2-3

##############[x,y,theta,v_n,v_0,omega_0,a_0,omega_dot_0]##################
TARGET_INIT = [-1000,500, math.pi/2, -0.3, 0.0, 0.0, 0.0] #SCENARIO FOR COMPARING Ts
TARGET_INIT = [-1000,-1000, math.pi/2, -0.3, 0.0, 0.0, 0.0] #SCENARIO VERY FAR
a = -0.3 #(m/s) increase for more amplitude of the "turn"
omega = +0.01#(-) #increase for faster sinusoidal beahviour

alpha_0, omega_0,alpha_dot_0,omega_dot_0 = TARGET_INIT[3],TARGET_INIT[4],TARGET_INIT[5],TARGET_INIT[6]

MAX_TARGET_VEL = 3 #(m/s) (only if target nTARGET_INIT = [-250,105, np.pi-np.pi/6, 0.2, 0.0, 0.0, 0.0] # SCENARIO 1o costant vels)
MIN_TARGET_VEL = 3 #(m/s)
sin_pattern = True

for i in range(len(AUV_XY)):
    AUV_XY[i,2] = math.atan2(TARGET_INIT[1]-AUV_XY[i,1],TARGET_INIT[0]-AUV_XY[i,0])

###################################################################################################################################


'''
|---- KINEMATIC SIMULATION: Initial AUV1 pose (m) --> [-73.27314635  30.71628413   1.93896648]
|---- KINEMATIC SIMULATION: Initial AUV2 pose (m) --> [ 0.82907326 47.07568912  2.65823081]
|---- KINEMATIC SIMULATION: Initial AUV3 pose (m) --> [ 61.00768948 -44.96615869   2.40857432]
|---- KINEMATIC SIMULATION: Initial Target(s) pose (m) --> [-100, 100, 2.6179938779914944]
'''