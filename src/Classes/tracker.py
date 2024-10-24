import os, pathlib, importlib.util

pkg_directory = os.path.dirname(pathlib.Path(__file__).parent.resolve())
class_path = pkg_directory+'/Classes'

spec = importlib.util.spec_from_file_location("module.est", class_path+"/estimation.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)

class Tracker:
    '''
    The Tracker class is created everytime we detect a target.
    It contains the entire state of the tracked object.
    '''
    def __init__(self,label):

        self.__estimator = est.Estimator()
        self.__curr_time = 0
        self.__prev_time = 0
        self.__label = label

    @property
    def state(self):
        return self.__estimator.current_estimate
    @property
    def regressor(self):
        return self.__estimator.current_regressor

    def processMeasurement(self, table): #table = [[tempo, misura, auv pos x, auv pos y]xTP]
        for i in range(len(table)):
            input_data = table[i]
            self.__estimator.iteration(input_data[0], input_data[1], input_data[2], input_data[3], self.__prev_time)

    def propagate_estimation(self, curr_time):
        self.__curr_time = curr_time
        self.__estimator.propagation(self.__curr_time, self.__prev_time)
        self.__prev_time = self.__curr_time


        
