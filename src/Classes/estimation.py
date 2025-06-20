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

import os, pathlib
import importlib.util
import numpy as np

# Import Config
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
class_path = pkg_directory + '/Classes'
spec = importlib.util.spec_from_file_location("module.config", class_path + "/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def state_vector_to_scalars(state_vector):
    return (state_vector[0, 0], state_vector[1, 0], state_vector[2, 0], state_vector[3, 0])

class Estimator:
    def __init__(self):

        self.recursiveEstimation = config.recursiveEstimation

        self.__x = np.zeros((4, 1))  # Initial state

        self.__P = np.eye(4) * 1000  # Initial large uncertainty
        self.__t = []
        self.__C = np.zeros((1, 4))
        self.t_prev = 0
        self.lambda_ = 0.99  # Forgetting factor
        self.P_max = config.P_max
        self.__damped = False
        self.__phi = []
        self.__y = []

    @property
    def current_estimate(self):
        return self.__x

    @property
    def current_covariance(self):
        if self.recursiveEstimation:
            return self.__P
        else:
            self.__P = self.computeCov()
        return self.__P

    @property
    def current_regressor(self):
        return (self.__phi, self.__y)

    def init_state_vector(self):
        return True

    def propagation(self, curr_time, prev_time):
        dt = curr_time - prev_time
        F = np.matrix([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        if self.recursiveEstimation == False:
            # Convert list of Jacobians (flattened) to matrix (N x 4)
            PHI = np.array(self.__phi)  # already a list of flat lists of length 4
            if PHI.ndim != 2 or PHI.shape[1] != 4:
                raise ValueError(f"Expected PHI to have shape (N, 4), got {PHI.shape}")

            # Convert list of measurements to column vector (N x 1)
            Y = np.array(self.__y).reshape(-1, 1)  # column vector

            # Batch least squares estimate
            self.__x = np.linalg.pinv(PHI) @ Y  # shape: (4, 1)
            self.__P = self.computeCov()
        self.__x = F @ self.__x
        self.__P = F @ self.__P @ F.T #+ np.eye(4) * 0.01  # small process noise
        print('self.__x',self.__x)

    def iterationRec(self, t_meas, y_i, si_x, si_y, prev_t):

       
        
        y_k = si_x * np.sin(y_i) - si_y * np.cos(y_i)
        delta_t = t_meas - prev_t if (t_meas - prev_t) != 0 else 1e-5  # avoid divide by zero

        phi_k = np.array([
                [np.sin(y_i)],
                [-np.cos(y_i)],
                [delta_t * np.sin(y_i)],
                [-delta_t * np.cos(y_i)]
            ])

        # Maintain logs
        self.__y.append(y_k)
        self.__t.append(t_meas)
        prev_t = self.__t[0]

        self.__phi.append(phi_k.flatten().tolist())

        for i in range(len(self.__phi)): # UPDATE REGRESSOR COLUMN
            if prev_t!=self.__t[i]:
                tmp = self.__phi[i]
                tmp[2] = ((self.__t[i]-self.__t[0])/(self.__t[i]-prev_t))*tmp[2]
                tmp[3] = ((self.__t[i]-self.__t[0])/(self.__t[i]-prev_t))*tmp[3]
                self.__phi[i] = [tmp[0],tmp[1],tmp[2],tmp[3]]

        # (trim if over length)
        if len(self.__y) > self.P_max:
            self.__y.pop(0)
            self.__t.pop(0)
            self.__phi.pop(0)

        # RLS gain
        phi_T = phi_k.T
        denom = self.lambda_ + phi_T @ self.__P @ phi_k
        K_k = (self.__P @ phi_k) / denom

        # Update state
        innovation = y_k - phi_T @ self.__x
        self.__x = self.__x + K_k * innovation

        # Update covariance
        self.__P = (1 / self.lambda_) * (self.__P - K_k @ phi_T @ self.__P)

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
        
        
    def computeCov(self, regularization=0.001, inflation=1.5):
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
        
        
        phi = np.array(self.__phi)

        y = np.array(self.__y)
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

    def iterationEKF(self, t_meas, y_i, si_x, si_y, prev_t):
        # TODO  for now this not work, problem with latencies
        # Measurement model: h(x) = si_x*sin(theta) - si_y*cos(theta)
        def h(x):
            px, py, vx, vy = x[0,0], x[1,0], x[2,0], x[3,0]
            # Example: predicted measurement = arctangent of relative position from sensor
            # or something related to state, not constants.
            return np.array([[np.arctan2(py - si_y, px - si_x)]])

        def H(x, delta_t):
            return np.array([
                [np.sin(y_i), -np.cos(y_i), delta_t * np.sin(y_i), -delta_t * np.cos(y_i)]
            ])

        y_k = np.array([[si_x * np.sin(y_i) - si_y * np.cos(y_i)]])
        delta_t = t_meas - prev_t if (t_meas - prev_t) != 0 else 1e-5
        H_k = H(self.__x, delta_t)

        # Innovation
        innovation = y_k - h(self.__x)

        # Kalman gain
        S = H_k @ self.__P @ H_k.T + np.eye(1) * 0.1
        K = self.__P @ H_k.T @ np.linalg.inv(S)

        # State update
        self.__x = self.__x + K @ innovation

        # Covariance update
        self.__P = (np.eye(4) - K @ H_k) @ self.__P

        # Maintain logs (for compatibility)
        self.__y.append(float(y_k))
        self.__t.append(t_meas)
        self.__phi.append(H_k.flatten().tolist())

        if len(self.__y) > self.P_max:
            self.__y.pop(0)
            self.__t.pop(0)
            self.__phi.pop(0)        
        

        

