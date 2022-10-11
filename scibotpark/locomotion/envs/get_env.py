from torchrl.env.vecenv import VecEnv
from torchrl.env.subproc_vecenv import SubProcVecEnv
from torchrl.env.base_wrapper import *
from torchrl.env.continuous_wrapper import *
from scibotpark.locomotion.envs.builder import build_locomotion_env
from gym.wrappers.time_limit import TimeLimit
import gym

DEFAULT_MAX_STEP = 1000

class NormObsWithImg(gym.ObservationWrapper, BaseWrapper):
  """
  Normalized Observation => Optional, Use Momentum
  """

  def __init__(self, env, epsilon=1e-4, clipob=10.):
    super(NormObsWithImg, self).__init__(env)
    self.count = epsilon
    self.clipob = clipob
    self._obs_normalizer = Normalizer(env.observation_space.shape)
    self.state_shape = np.prod(env.observation_space.shape)

  def copy_state(self, source_env):
    # self._obs_rms = copy.deepcopy(source_env._obs_rms)
    self._obs_var = copy.deepcopy(source_env._obs_var)
    self._obs_mean = copy.deepcopy(source_env._obs_mean)

  def observation(self, observation):
    if self.training:
      self._obs_normalizer.update_estimate(
        observation[..., :self.state_shape]
      )
    img_obs = observation[..., self.state_shape:]
    return np.hstack([
      self._obs_normalizer.filt(observation[..., :self.state_shape]),
      img_obs
    ])

def get_visual_env(env_id, env_param):
  env = get_single_env(env_id, env_param)
  if "obs_norm" in env_param and env_param["obs_norm"]:
    if env_param['env_build']['env_kwargs']['include_vision_obs']:
        if env_param['env_build']['env_kwargs']['include_vision_obs']:
            env = NormObsWithImg(env)
        else:
            env = NormObs(env)

  return env

def get_single_env(env_id, env_param):
    # print(env_id, env_param)
    env = build_locomotion_env(**env_param['env_build'])
    env = BaseWrapper(env)

    # add horizon limitation9
    if 'horizon' not in env_param:
        env = TimeLimit(env = env, max_episode_steps = DEFAULT_MAX_STEP)
    else:
        env = TimeLimit(env = env, max_episode_steps = env_param['horizon'])
    
    if isinstance(env.action_space, gym.spaces.Box):
        env = NormAct(env)
    return env

def get_subprocvec_env(env_id, env_param, vec_env_nums, proc_nums):
    """
    return a list of envs of a single env based on the multiprocess configuration
    """
    if isinstance(env_param, list):
        assert vec_env_nums % len(env_param) == 0 
        env_args = [
            [env_id, env_sub_params] for env_sub_params in env_param
        ] * (vec_env_nums // len(env_param))

        vec_env = SubProcVecEnv(
            proc_nums, vec_env_nums, [get_single_env] * vec_env_nums, env_args
        )

        if "obs_norm" in env_param[0] and env_param[0]["obs_norm"]:
            if env_param[0]['env_build']['env_kwargs']['include_vision_obs']:
                vec_env = NormObsWithImg(vec_env)
            else:
                vec_env = NormObs(vec_env)
    
    else:
        vec_env = SubProcVecEnv(
            proc_nums, vec_env_nums, get_single_env, [env_id, env_param]
        )

        if "obs_norm" in env_param and env_param["obs_norm"]:
            if env_param['env_build']['env_kwargs']['include_vision_obs']:
                if env_param['env_build']['env_kwargs']['include_vision_obs']:
                    vec_env = NormObsWithImg(vec_env)
                else:
                    vec_env = NormObs(vec_env)

    return vec_env

