import numpy as np

class Task(object):
    def __init__(self):
        """task initialize"""
        self.current_pos=np.zeros(0)
        self.last_pos=np.zeros(0)

    def reset(self,env):
        self._env=env
    
    def update(self,env):
        inertial_data=env.robot.get_inertial_data()
        
        self.last_pos=self.current_pos
        self.current_pos=inertial_data["position"]

    def compute_reward(self,env):
        pass

    def done(self,env):
        pass  