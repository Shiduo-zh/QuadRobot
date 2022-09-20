import gym
import numpy as np

class DeltaActionRestrainEnv(gym.ActionWrapper):
  def __init__(self,env,action_clip):
    super().__init__(env)
    self.clip_num = action_clip
    if isinstance(self.clip_num,list):
      self.clip_num = np.array(self.clip_num)
    self.ub = np.zeros(*self.clip_num.shape)+self.clip_num
    self.lb = np.zeros(*self.clip_num.shape)-self.clip_num
    self.action_sapce = gym.spaces.Box(self.lb,self.ub)
  
  def action(self,action):
    return np.clip(action,self.lb,self.ub)

class DiagonalActionEnv(gym.ActionWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.lb = np.split(self.env.action_space.low, 2)[0]
    self.ub = np.split(self.env.action_space.high, 2)[0]
    self.action_space = gym.spaces.Box(self.lb, self.ub)

  def action(self, action):
    right_act, left_act = np.split(action, 2)
    act = np.concatenate(
      [right_act, left_act, left_act, right_act]
    )
    return act

