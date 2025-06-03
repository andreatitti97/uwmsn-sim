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

def computePursuitVel(curr_est,s_pose,d_max, DT):

    predicted_pose = np.array(np.zeros(2))
    predicted_pose[0] = curr_est[0,0] + DT*curr_est[2,0]#TODO: check how much time propagate
    predicted_pose[1] = curr_est[1,0] + DT*curr_est[3,0]

    eucl_dist = np.sqrt((curr_est[0,0]-s_pose[0])**2+(curr_est[1,0]-s_pose[1])**2)
    epsi = 0#config.RANGE_TO_TARGET #DISTANZA VOLUTA DAL TARGET
    
    beta = 1/d_max #coeficente angolare retta per due punti m = y2-y1/x1-x2 

    x = (eucl_dist-epsi)
    v_n = beta*x + np.sqrt((curr_est[2,0])**2+(curr_est[3,0])**2)#feed forward the estimated velocity of the target

    if v_n > config.AUV_MAX_VEL:
        v_n = config.AUV_MAX_VEL

    return v_n

def computeCost(phi):

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

# Utils functions
def computeCost_fi(phi):

    length_y = len(phi)
    W = np.zeros((length_y,length_y))
    for i in range(length_y):
        W[i,i] = 1.0
    PHI = np.dot(np.transpose(phi),np.dot(np.linalg.inv(W),phi))

    return np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)

def computeCov2(y,phi):

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

def computeCov(y, phi, regularization=0.001, inflation=1.5):
    """
    Compute the posterior covariance matrix for the estimator.

    Parameters:
    - y: list or array of measurements
    - phi: 2D array (regressor), shape (n_samples, n_features)
    - sigma_meas: standard deviation of measurement noise
    - regularization: optional float for Tikhonov regularization (lambda)
    - inflation: optional float to inflate the covariance (e.g., 2.0)

    Returns:
    - cov: covariance matrix (shape: n_features x n_features)
    """
    
    phi = np.array(phi)

    y = np.array(y)
    n = len(y)

    # Compute measurement covariance (variance on diagonal)
    R_inv = np.identity(n) / (config.SIGMA_MEAS ** 2)

    # Optional regularization
    if regularization is not None:
        cov_inv = phi.T @ R_inv @ phi + regularization * np.identity(phi.shape[1])
    else:
        cov_inv = phi.T @ R_inv @ phi

    cov = np.linalg.inv(cov_inv)

    # Optional inflation
    if inflation is not None:
        cov *= inflation

    return cov


def damped_pseudoinverse_full(phi_list, lambd=0.01, s_star=0.001):
    """
    Computes the damped pseudoinverse φ^# of a tall matrix φ ∈ R^{m×n}
    using full SVD and Tikhonov regularization.
    
    Returns a matrix φ^# ∈ R^{n×m}.
    """
    phi = np.array(phi_list)
    m, n = phi.shape
    U, S, VT = np.linalg.svd(phi, full_matrices=True)  # full SVD: U ∈ R^{m×m}, V ∈ R^{n×n}
    
    # Construct Σ̃⁻¹ (damped inverse with padding)
    S_damped_inv = np.zeros((n, m))  # shape (n x m), matching φ^#
    for i in range(len(S)):
        s_i = S[i]
        if s_i >= s_star:
            lambda_i =  0
        else:
            # Apply Tikhonov regularization
            lambda_i = lambd(0.5*np.cos(s_i*np.pi/s_star)+0.5)
            # If singular value is zero, use the regularization parameter        
        S_damped_inv[i, i] = s_i / s_i**2+lambda_i**2
    
    # Compute φ^# = V * S_damped_inv * U.T
    print("Shape of U:", U.shape)
    print("Shape of S_damped_inv:", S_damped_inv.shape)
    print("Shape of VT:", VT.shape)
    phi_pinv = VT @ S_damped_inv @ U.T
    return phi_pinv.tolist()

# Weighted distance metric (ETC trigger)
def weighted_distance(seq1, seq2, alpha=0.8):
    w = np.array([alpha**(h) for h in range(len(seq1))])
    d_i = 0.0
    
    for i in range(len(seq1)):
        tmp_seq1 = np.array(seq1[i])
        tmp_seq2 = np.array(seq2[i])
        d_i += w[i]*np.linalg.norm(tmp_seq1 - tmp_seq2)
    
    return np.sum(d_i)

# Simple function that implements Newton-eulero for the underactuated AUV
def systemModel(senPose, U, H, dt):

    s = [senPose[0], senPose[1], senPose[2]] # [x,y,theta]
    # Initialize the list to store the predicted states
    s_hat = []
    # Compute new headingRef according to the given heading change
    for i in range(H+1):

        s[2] = s[2]+(U[i])       
        s[0] = s[0]+np.cos(s[2])*U[H+1+i]*(dt)#TODO: FIX THIS PROBLEM OF THE INDEX for plcy intent
        s[1] = s[1]+np.sin(s[2])*U[H+1+i]*(dt)
        s_hat.append([s[0],s[1]])

    return s_hat

def kinematic_control_auv(s, x_des, y_des, theta_ref, K_p, dt, e_ij):
    """
    Implements kinematic control for an underactuated AUV.
    
    Parameters:
    - s: np.array([x, y, theta]) -> Current state (position and heading)
    - x_des: float -> Desired x position
    - y_des: float -> Desired y position
    - theta_ref: float -> Reference desired heading
    - e_ij: list or np.array -> Errors with neighboring agents
    - K_p: float -> Proportional gain for coordination term
    - dt: float -> Time step for integration
    
    Returns:
    - s_next: np.array([x_next, y_next, theta_next]) -> Updated state
    - u: float -> Controlled surge velocity
    - r: float -> Yaw rate control input
    """
    # Compute desired surge velocity
    dx = x_des - s[0]
    dy = y_des - s[1]
    u_ref = np.sqrt(dx**2 + dy**2) / dt  # Compute reference surge velocity
    
    # Compute controlled surge velocity
    u = u_ref - K_p * np.sum(e_ij)
    
    # Compute yaw rate (simple proportional control to follow reference heading)
    r = (theta_ref - s[2]) / dt  # Approximating derivative as finite difference
    
    # Apply kinematic model
    x, y, theta = s
    x_next = x + u * np.cos(theta) * dt
    y_next = y + u * np.sin(theta) * dt
    theta_next = theta + r * dt
    return np.array([x_next, y_next, theta_next]), u, r
