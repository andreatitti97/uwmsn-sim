import importlib, pathlib
import numpy as np
import matplotlib.pyplot as plt
# Environment: Define the relevant paths
class_directory = pathlib.Path(__file__).parent.resolve()

spec = importlib.util.spec_from_file_location("module.config", class_directory/'config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def calc_spline_course(sp,ds):
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk, surge = [], [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        iu = sp.calc_surge(i_s)
        rx.append(ix)
        ry.append(iy)
        surge.append(iu)
        
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s, surge

def computePursuitVel(curr_est,s_pose,d_max):

    predicted_pose = np.array(np.zeros(2))
    predicted_pose[0] = curr_est[0,0] + config.DT*curr_est[2,0]#TODO: check how much time propagate
    predicted_pose[1] = curr_est[1,0] + config.DT*curr_est[3,0]

    eucl_dist = np.sqrt((curr_est[0,0]-s_pose[0])**2+(curr_est[1,0]-s_pose[1])**2)
    epsi = config.RANGE_TO_TARGET #DISTANZA VOLUTA DAL TARGET
    
    beta = 1/d_max #coeficente angolare retta per due punti m = y2-y1/x1-x2 

    x = (eucl_dist-epsi)
    v_n = beta*x + np.sqrt((curr_est[2,0])**2+(curr_est[3,0])**2)

    if v_n > config.AUV_MAX_VEL:
        v_n = config.AUV_MAX_VEL

    return v_n


def compute_cost(phi):

    length_y = len(phi)
    tmp_phi = np.zeros((length_y,2))
    for i in range(length_y):
        a = phi[i]
        tmp_phi[i,:] = [a[0],a[1]]
    
    W = np.zeros((length_y,length_y))
    for i in range(length_y):
        W[i,i] = 1.0

    PHI = np.dot(np.transpose(tmp_phi),np.dot(np.linalg.inv(W),tmp_phi))

    return np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)

def computeCov(y,phi):

    # Compute Covariance of the target state
    R = np.zeros((len(y),len(y))) #matrice diagonale perchè errori sulle singole misure indipendenti tra loro            
    for i in range(len(y)): 
        for j in range(len(y)):
            if i == j:
                R[i,j] = (config.SIGMA_MEAS)
            else:
                R[i,j] = 0 

    a = config.SIGMA_MEAS
    cov = np.linalg.inv(np.dot(np.dot(np.transpose(phi),np.linalg.inv(a*np.identity(len(y)))),phi))
        
    return cov