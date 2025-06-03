#Import basic system modules
import os, pathlib
import importlib.util
import numpy as np

# Import Costum classes
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
class_path = pkg_directory+'/Classes'

spec = importlib.util.spec_from_file_location("module.config", class_path+"/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def state_vector_to_scalars(state_vector):
    '''
    Returns the elements from the state_vector as a tuple of scalars.
    '''
    return (state_vector[0][0,0],state_vector[1][0,0],state_vector[2][0,0],state_vector[3][0,0])    
    
def damped_pseudoinverse_full(phi_list, lambd=0.01, s_star=0.001):
    """
    Computes the damped pseudoinverse φ^# of a tall matrix φ ∈ R^{m×n}
    using full SVD and Tikhonov regularization.
    
    Returns a matrix φ^# ∈ R^{n×m}.
    """
    phi = np.array(phi_list)
    m, n = phi.shape
    U, S, VT = np.linalg.svd(phi, full_matrices=True)  # full SVD: U ∈ R^{m×m}, V ∈ R^{n×n}
    
    lambd = 0.0001
    s_star = 0.00001 * np.max(S)
    """lambd (low) Less damping, more sensitive to noise (high) More damping, smoother but may oversmooth
    s_star	(low) Fewer singular values regularized	(high) More values regularized, even moderately small ones"""

    # Construct Σ̃⁻¹ (damped inverse with padding)
    S_damped_inv = np.zeros((n, m))  # shape (n x m), matching φ^#
    for i in range(len(S)):
        s_i = S[i]
        if s_i >= s_star:
            lambda_i =  0
        else:
            # Apply Tikhonov regularization
            lambda_i = lambd*(0.5*np.cos(s_i*np.pi/s_star)+0.5)
            # If singular value is zero, use the regularization parameter        
        S_damped_inv[i, i] = s_i / (s_i**2 + lambda_i**2)
    
    # Compute φ^# = V * S_damped_inv * U.T
    phi_pinv = VT @ S_damped_inv @ U.T
    return phi_pinv.tolist()

class Estimator:
    def __init__(self):
        '''
        Each object being tracked will result in the creation of a new ExtendedKalmanFilter instance.
        '''
        self.__x = []
        self.__phi = []
        self.__y = []
        self.__C = np.zeros((1,4))
        self.__t = []
        self.__damped = False
        self.t_prev = 0
        self.P_max = config.P_max

    @property
    def current_estimate(self):
        return (self.__x)
    @property
    def current_regressor(self):
        return (self.__phi, self.__y)

    def init_state_vector(self):
        return True

    def propagation(self, curr_time, prev_time):#compute old state in the regressor and propagation to the actual state
        # Invert the equations for computing the state at the oldest time in the regressor
        tmp_y = np.zeros((len(self.__y),1))
        for i in range(len(self.__y)):
            tmp_y[i] = self.__y[i]

        if self.__damped:
            phi_pinv_tilde = damped_pseudoinverse_full(self.current_regressor[0])
            self.__x = np.dot(phi_pinv_tilde,tmp_y)
        else:
            self.__x = np.dot(np.linalg.pinv(self.__phi),tmp_y)
        
        # Propagate the estimation
        dt = curr_time - self.__t[0] #tempo attuale - tempo ultimo stato noto.
        self.__F = np.matrix([[1,0,dt,0],
                              [0,1,0,dt],
                              [0,0,1,0],
                              [0,0,0,1]])
        self.__x = self.__F*self.__x

    def iteration(self, t_meas, y_i, si_x, si_y, prev_t):
        
        self.__y.append(si_x*np.sin(y_i) - si_y*np.cos(y_i))
        self.__t.append(t_meas)
        prev_t = self.__t[0]
        self.__C = [np.sin(y_i), -np.cos(y_i), (t_meas - self.__t[0])*np.sin(y_i), -(t_meas - self.__t[0])*np.cos(y_i)]
        self.__phi.append(self.__C)

        if len(self.__y) == self.P_max:
            self.__y.pop(0) #SHIFT
            self.__t.pop(0)
            self.__phi.pop(0)          
        for i in range(len(self.__phi)): # UPDATE REGRESSOR COLUMN
            if prev_t!=self.__t[i]:
                tmp = self.__phi[i]
                tmp[2] = ((self.__t[i]-self.__t[0])/(self.__t[i]-prev_t))*tmp[2]
                tmp[3] = ((self.__t[i]-self.__t[0])/(self.__t[i]-prev_t))*tmp[3]
                self.__phi[i] = [tmp[0],tmp[1],tmp[2],tmp[3]]
        


        
        

        

