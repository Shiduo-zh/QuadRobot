from scibotpark.locomotion.envs.task.task import Task
import numpy as np

class ForwardTask(Task):
    def __init__(self,
                forward_reward_ratio=1,
                alive_reward_ratio=0.1,
                torque_reward_ratio=0.05,
                heading_reward_ratio=0.1,
                fall_reward=0,
                z_constrain=True,
                z_penalty=0.1,
                alive_height_range=[0.15,0.35],
                alive_roll_limit=np.pi/4,
                subgoal_reward=None,
                target_velocity=None,
                ):
        """
        params:
            forward_reward_ratio:the reward weight for moving forward velocity,
            alive_reward_ratio: the reward weight for alive in one step,
            torque_reward_ratio: the penalty weight for much energy consume,
            heading_reward_ratio: the penalty weight for unreseanable orientation(too high or too low),
            fall_reward:the penalty for the fall of the agent in one episode
            alive_height_range: the range of living heights, to judge if one episode is done
            alive_roll_limit: the upper bound of the rotation restriction
            subgoal_reward: extra bonus weight,
            target_velocity: expected velocity
        
        """
        super(ForwardTask,self).__init__()
        #reward weight settings
        self.forward_weight = forward_reward_ratio
        self.alive_weight = alive_reward_ratio
        self.energy_weight = torque_reward_ratio
        self.heading_weight = heading_reward_ratio
        self.fall_reward = fall_reward
        self.is_fall=False
        self.subgoad_reward = subgoal_reward
        self.z_penalty = z_penalty

        #restrictions
        self.alive_roll_limit = alive_roll_limit
        self.alive_height_range = alive_height_range
        self.z_constrain = z_constrain
    
    def compute_reward(self,env):
        # del env

        info={}
        forward_reward = self.forward_weight*self._compute_forward_reward(env)
        energy_reward = self.energy_weight*self._compute_energy_reward(env)
        alive_reward = self.alive_weight
        heading_reward = self.heading_weight*self._compute_energy_reward(env)

        reward = forward_reward+energy_reward+alive_reward+heading_reward
        
        done = self.done(env)
        if self.is_fall:
            reward += self.fall_reward

        if self.subgoad_reward is not None:
            pass

        info['alive_reward'] = alive_reward
        info['energy_reward'] = energy_reward
        info['forward_reward'] = forward_reward
        info['heading_reward'] = heading_reward
        
        return reward, info, done
    
    def _compute_forward_reward(self,env):
        # del env
        inertial_data=env.robot.get_inertial_data()
        reward=inertial_data["linear_velocity"][0]
        if self.z_constrain:
            reward-=self.z_penalty*inertial_data["linear_velocity"][2]
        
        return reward

    def _compute_heading_rerward(self,env):
        # del env
        inertial_data=env.robot.get_inertial_data()
        roll,pitch,yaw=inertial_data["rotation"]
        penalty=-np.power(pitch,2)
        return penalty
        
    def _compute_energy_reward(self,env):
        # del env
        torques=env.robot.get_joint_states("torque")
        return -np.power(np.linalg.norm(torques),2)
    
        
    def done(self,env):
        # del env
        inertial_data=env.robot.get_inertial_data()
        roll,pitch,yaw=inertial_data["rotation"]
        height=inertial_data["position"][2]
        if roll > self.alive_roll_limit or height < self.alive_height_range[0] or height > self.alive_height_range[1] : 
            self.is_fall = True
            return True
        elif env.step_num > env.horizon:
            self.is_fall = False
            return True
        else:
            self.is_fall = False
            return False




        