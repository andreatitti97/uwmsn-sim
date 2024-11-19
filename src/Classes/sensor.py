from math import atan2, pi
import numpy as np

class Sensor:
    """ Simulate Vector Sensor """
    
    def __init__(self, name, f, mean, variance):
        self.name = name
        self.f = f
        self.mean = mean
        self.variance = variance


        self.abs_bearing = 0

        # For controlling outlier behavior
        self.measurement_count = 0
        self.outlier_count = 0
        self.reset_size = 100  # Reset after 100 measurements
        self.outlier_percentage = 10.0

    def setOutlierPercentage(self, percentage, reset_size=100):
        """
        Set the percentage of outliers for the sensor measurements.
        
        Parameters:
        - percentage: Float in range [0, 1], the fraction of measurements that should be outliers.
        - reset_size: Number of measurements after which the counts reset.
        """
        self.outlier_percentage = percentage
        self.reset_size = reset_size
        self.measurement_count = 0
        self.outlier_count = 0

    def measureBearing(self, xt, yt, obs_pos, orientation):
        vect = [xt - obs_pos[0], yt - obs_pos[1]]
        self.abs_bearing = atan2(vect[1], vect[0])  # abs bearing = rel_bearing - vehicle ori -> [-pi,+pi]
        
        if orientation < 0:
            theta_tmp = 2 * pi + orientation
        else:
            theta_tmp = orientation

        if self.abs_bearing < 0:
            self.abs_bearing = 2 * pi + self.abs_bearing  # Change convention Bearing_abs -> [0,2*pi]
        
        if theta_tmp <= self.abs_bearing:  # If the target is counterclockwise w.r.t. surge vel
            rel_bearing = self.abs_bearing - theta_tmp 
        else:
            rel_bearing = 2 * pi - (theta_tmp - self.abs_bearing)

        # Create the noise and add it to the measurement
        self.noise = self.mean + np.random.uniform(-self.variance, self.variance)
        self.abs_bearing = self.abs_bearing + self.noise  # Overwrite absolute bearing with corrupted quantities

        return self.abs_bearing, rel_bearing, obs_pos

    def measureWithOutliers(self, xt, yt, obs_pos, orientation, outlier_range=(0, pi)):
        """
        Measure the bearing with a controlled percentage of outliers.

        Parameters:
        - xt, yt: Target position coordinates.
        - obs_pos: Observer position coordinates.
        - orientation: Orientation of the observer.
        - outlier_range: Range for generating outliers (default [0, 2π]).

        Returns:
        - Absolute bearing (with noise or as an outlier).
        - Relative bearing.
        - Observer position.
        """
        # Increment measurement count
        self.measurement_count += 1
        
        # Determine if this measurement should be an outlier
        is_outlier = False
        if self.outlier_count / self.measurement_count < self.outlier_percentage:
            is_outlier = np.random.rand() < self.outlier_percentage

        if is_outlier:
            # Generate outlier
            self.outlier_count += 1
            abs_bearing_outlier = np.random.uniform(outlier_range[0], outlier_range[1])
            rel_bearing_outlier = abs_bearing_outlier  # Simplified, can adjust for observer orientation
            result = (abs_bearing_outlier, rel_bearing_outlier, obs_pos)
        else:
            # Use normal measurement
            result = self.measureBearing(xt, yt, obs_pos, orientation)

        # Reset counts if necessary
        if self.measurement_count >= self.reset_size:
            self.measurement_count = 0
            self.outlier_count = 0

        return result
