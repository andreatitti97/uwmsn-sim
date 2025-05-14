import os, pathlib, importlib
import numpy as np
# Import costum modules
pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
class_path = pkg_directory+'/Classes'

spec = importlib.util.spec_from_file_location("module.config", class_path+"/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
sin_pattern = config.sin_pattern

class Pose:
    """2D pose"""

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class Target:
    """
    Constructs an instantiate the AUV

    Parameters
    ----------
    max_linear_speed : (float)
        The maximum linear speed that the robot can go
    max_angular_speed : (float)
        The maximum angular speed that the robot can rotate about its vertical
        axis
    controller : (Controller)
        A configurable controller to finds the path and calculates command
        linear and angular velocities. 
    """

    def __init__(self, label, exist):

        if label == 1:
            self.pose = Pose(config.TARGET_INIT[0], config.TARGET_INIT[1],  config.TARGET_INIT[2])
        elif label == 2:
            self.pose = Pose(-70, +70,  0)
        elif label == 3:
            self.pose = Pose(-50, -50,  0)
        self.lin_vel = config.alpha_0 #(m/s)
        self.ang_vel = config.omega_0 #(rad/sec)
        self.lin_acc = config.alpha_dot_0 #(m/sec^2)
        self.ang_acc = config.omega_dot_0 #(rad/sec^2)
        self.exist = exist
        self.label = label
        self.t = 0
        if sin_pattern == True:
            self.a = config.a#(m/s)
            self.omega = config.omega #(-)


    def set_start_target_poses(self, pose):
        """
        Sets the start and target positions of the robot

        Parameters
        ----------
        pose_start : (Pose)
            Start postion of the robot (see the Pose class)
        pose : (Pose)
            Target postion of the robot (see the Pose class)
        """
        self.pose = pose


    def move_target(self, dt):
        """
        Moves the target for one time step increment

        Parameters
        ----------
        dt : (float)
            time step
        """
        self.t += dt
        self.pose.theta = self.pose.theta + (self.ang_vel+self.ang_acc * dt)*dt 
        if sin_pattern == True:
            #self.pose.x = self.pose.x + self.a*np.sin(self.omega*self.t+np.pi)*dt
            #self.pose.y = self.pose.y + self.lin_vel*dt
            self.pose.y = self.pose.y + self.a*np.cos(self.omega*self.t+np.pi)*dt
            self.pose.x = self.pose.x + self.lin_vel*dt
        else:
            self.pose.x = self.pose.x + (self.lin_vel+self.lin_acc*dt) * \
                np.cos(self.pose.theta) * dt 
        
            self.pose.y = self.pose.y + (self.lin_vel+self.lin_acc*dt) * \
                np.sin(self.pose.theta) * dt 
        
        