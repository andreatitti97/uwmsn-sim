#!/usr/bin/env python
#Import basic system modules
import os
import importlib.util, pathlib

# Import Costum classes
class_path = pathlib.Path(__file__).parent.resolve()
class_path = os.path.dirname(os.path.dirname(class_path))
class_path = class_path+'/src'+'/Classes'

spec = importlib.util.spec_from_file_location("module.config", class_path+'/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
spec = importlib.util.spec_from_file_location("module.sensor", class_path+'/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("module.sensor", class_path+'/sensor.py')
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)
spec = importlib.util.spec_from_file_location("module.tracker", class_path+'/tracker.py')
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)

spec = importlib.util.spec_from_file_location("module.target", class_path+'/target.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


class_path = os.path.dirname(os.path.dirname(os.path.dirname(class_path)))+'/uwmsn-motion_opt'+'/src'+'/Classes'
spec = importlib.util.spec_from_file_location("module.estimator", class_path+'/estimator.py')
estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator_module)