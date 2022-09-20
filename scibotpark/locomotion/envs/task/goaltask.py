from scibotpark.locomotion.envs.task.task import Task
import numpy as np

class GoalTask(Task):
    def __init__(self,
                forward_reward_ratio=1,
                alive_reward_ratio=0.1,
                torque_reward_ratio=0.05,
                goal_reward_ratio=0.5,
                goal_bonus=10,
                end_dist=0.2,
                fall_reward=0,
                target_velocity=None,
                goal_position=(1,1,1),
                timestep=1./500,
                ):
        """
        param:
            end_dist:if the distance between goal and agent is less than end_dist, 
                     it will get the bonus of the goal
        """
        super(GoalTask,self).__init__()
        
        self.goal_position=goal_position
        self.timestep=timestep
        #reward settings
        self.forward_weight=forward_reward_ratio
        self.alive_weight=alive_reward_ratio
        self.energy_weight=torque_reward_ratio
        self.goal_weight=goal_reward_ratio
        self.fall_reward=fall_reward
        self.goal_reward=goal_bonus
        
        #restrictions
        self.end_dist=end_dist
      
    def compute_reward(self, env):
        del env

        info={}
        energy_reward=self.energy_weight*self._compute_energy_reward(env)
        forward_reward=self.forward_weight*self._compute_forward_reward(env)
        goal_reward=self.goal_weight*self._compute_goal_reward(env)
        alive_reward=self.alive_weight

        reward=energy_reward+forward_reward+goal_reward+alive_reward
        
        if self.done(env):
            reward+=self.fall_reward
        
        if np.linalg.norm(self.current_pos-self.goal_position)<self.end_dist:
            reward+=self.goal_reward
        
        info['alive_reward'] = alive_reward
        info['energy_reward'] = energy_reward
        info['forward_reward'] = forward_reward
        info['goal_reward'] = goal_reward
        
        return reward

    def _compute_energy_reward(self,env):
        del env
        torques=env.robot.get_joint_states("torque")
        return -np.power(np.linalg.norm(torques),2)
    
    def _compute_forward_reward(self,env):
        del env
        inertial_data=env.robot.get_inertial_data()
        velocity=inertial_data["linear_velocity"]
        target_dir=self.goal_position-self.current_pos
        reward=np.dot(velocity,target_dir)/np.linalg.norm(target_dir)
        return reward

    def _compute_goal_reward(self,env):
        del env
        last_dist=np.linalg.norm(
            np.array(self.goal_position)-self.last_pos
        )
        current_dist=np.linalg.norm(
            np.array(self.goal_position)-self.current_pos
        )
        reward=(last_dist-current_dist)/self.timestep
        return reward
    
    def done(self,env):
        pass
        



