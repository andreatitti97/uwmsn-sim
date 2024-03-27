import importlib, pathlib
import numpy as np

# Environment: Define the relevant paths
class_directory = pathlib.Path(__file__).parent.resolve()

spec = importlib.util.spec_from_file_location("module.config", class_directory/'config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def calc_spline_course(sp,ds):
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
    return rx, ry, ryaw, rk, s

def computePursuitVel(curr_est,s_pose,d_max):

    predicted_pose = np.array(np.zeros(2))
    predicted_pose[0] = curr_est[0,0] + config.DT*curr_est[2]
    predicted_pose[1] = curr_est[1,0] + config.DT*curr_est[3]

    eucl_dist = np.sqrt((predicted_pose[0]-s_pose[0])**2+(predicted_pose[1]-s_pose[1])**2)
    epsi = config.RANGE_TO_TARGET #DISTANCA VOLUTA DAL TARGET
    
    alpha = 0.09
    x = (eucl_dist-epsi)
    
    weigth = 1/(1 + np.exp(alpha*(-x+d_max/2)))
    v_n = weigth*config.AUV_MAX_VEL

    if x < 1:
        v_n = -10**3

    return v_n

def compute_cost(phi,len_y):

    tmp_phi = np.zeros((len_y,4))
    for i in range(len_y):
        row = phi[i]
        tmp_phi[i,:] = [row[0],row[1],row[2],row[3]]
    PHI = np.dot(np.transpose(tmp_phi[:,0:2]),tmp_phi[:,0:2]) 
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