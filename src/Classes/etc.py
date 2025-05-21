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

spec = importlib.util.spec_from_file_location("module.utils", class_path+"/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class EventHandler:
    def __init__(self, targetNum, H, DT):
        

        #Consensus Routine init
        self.targetNum = targetNum
        self.initEtcEstimation = False
        self.decisionConsensus = False
        self.cov_tilde = [np.matrix([[0],[0],[0],[0]]) for _ in range(targetNum)]
        self.xi_hat_tilde = [[] for _ in range(targetNum)]
        self.t_est = [0.0 for _ in range(targetNum)]
        self.KLD_thres = 25

        # Guidance Routine init
        self.initEtcGuidance = False
        self.decisionGuidance = False
        self.H = H
        self.DT = DT
        self.beta = config.alphaETC
        self.delta_0 = 50
        self.pos_scale = 10.0
        self.vel_scale = 0.1
        self.t_last = 0.0

    def etcRoutineConsensus(self,t,targetIndex,x_hat,cov):
   
        # ETC mechanism
        if not self.initEtcEstimation:
            self.cov_tilde[targetIndex] = cov
            self.xi_hat_tilde[targetIndex] = x_hat
            self.t_est[targetIndex] = t
            self.initEtcEstimation = True
        else:
            delta_t = t - self.t_est[targetIndex]
            F = np.array([
                [1, 0, delta_t, 0],
                [0, 1, 0, delta_t],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            self.xi_hat_tilde[targetIndex] = F @ self.xi_hat_tilde[targetIndex]
            self.cov_tilde[targetIndex] = F @ self.cov_tilde[targetIndex] @ F.T

            kld_trace = np.trace(np.linalg.inv(cov) @ self.cov_tilde[targetIndex] - np.eye(4))
            kld_logdet = np.log(np.linalg.det(cov) / np.linalg.det(self.cov_tilde[targetIndex]))
            delta = self.xi_hat_tilde[targetIndex] - x_hat
            delta[0:2] /= self.pos_scale   # e.g., pos_scale = 10 meters
            delta[2:4] /= self.vel_scale   # e.g., vel_scale = 1 m/s
            kld_norm = np.linalg.norm(delta)
            KLD = 0.5 * (kld_trace + kld_norm + kld_logdet + 4)
            print('DEBUG ETC ROUTINE: KLD estimated target state',KLD)
            print('Old State',self.xi_hat_tilde[targetIndex])
            print('New State',x_hat)
            if KLD > self.KLD_thres:
                self.cov_tilde[targetIndex] = cov
                self.xi_hat_tilde[targetIndex] = x_hat
                self.decisionConsensus = True
            else:
                self.t_est[targetIndex] = t
    
    def etcRoutineGuidance(self,t,senPose,ctrlPolicy):

        
        if self.initEtcGuidance == False:

            self.U_k_1 = ctrlPolicy[3:]
            self.initEtcGuidance = True

        else:
            U_k = ctrlPolicy[3:]
            print('U_k',U_k)
            print('U_k_1',self.U_k_1)
            s = [senPose[0],senPose[1],senPose[2]] # [x,y,theta]
            bar_s_hat_k = utils.systemModel(s,U_k,self.H,self.DT)
            bar_s_hat_k_1 = utils.systemModel(s,self.U_k_1,self.H,self.DT)
            dist = utils.weighted_distance(bar_s_hat_k,bar_s_hat_k_1)
            # Adaptive threshold
            
            print('bar_s_hat_k',bar_s_hat_k)
            print('bar_s_hat_k_1',bar_s_hat_k_1)
            print('dist',dist)
            print('delta_t',(t-self.t_last))
            delta_k = self.delta_0*np.exp(-self.beta*(t-self.t_last))
            print('delta_k',delta_k)
            # Trigger check
            if dist > delta_k:

                self.t_last = t 
                self.decisionGuidance = True

            # Shift in time the control sequence + add heuristic
            self.U_k_1 = [ctrlPolicy[4],ctrlPolicy[5],ctrlPolicy[6],0.0,
                    ctrlPolicy[8],ctrlPolicy[9],ctrlPolicy[10],ctrlPolicy[10]]
            
