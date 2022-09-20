import numpy as np
import gym
import pybullet as p
from pybullet_utils import bullet_client
from scibotpark.locomotion.envs.envs import UnitreeLocomotionEnv


from scibotpark.locomotion.envs.task import goaltask, forwardtask

from scibotpark.locomotion.utils import get_default_randomization
from scibotpark.locomotion.envs.envs import UnitreeLocomotionEnv
from scibotpark.locomotion.envs.wrapper import DeltaActionRestrainEnv, DiagonalActionEnv
from scibotpark.locomotion.envs.terrains.terrains import getTerrainCls

def build_locomotion_env(
    goal:bool,
    rand:bool,
    diagnal:bool,
    terrain:str,
    action_clip = None,
    task_kwargs = dict(),
    env_kwargs = dict(),
    **kwargs
):
    """
    parameters:
    goal: wether a goal task or forward task
    rand: if take domain randomization per epoch
    step_rand: if take externel force on agent  
    diagnal: if strict the action into diagnal (12 dof to 6 dof)
    terrain: the terrains and obstacle settings
    task_kwargs: a dict for task configuration
    env_kwargs: a dict for basic locomotion environment configuration
    """
    pb_client= bullet_client.BulletClient(connection_mode= p.GUI)  
    # goal setting
    if goal:
        task = goaltask.GoalTask(**task_kwargs)
    else:
        task = forwardtask.ForwardTask(**task_kwargs)
    
    # domain randomization configuration
    if rand:
        domain_randomization = get_default_randomization()

    TerrainCls, _ = getTerrainCls(terrain)
    surrounding = TerrainCls(pb_client)

    env = UnitreeLocomotionEnv(
        task = task,
        surrounding = surrounding,
        domain_randomization = domain_randomization,
        pb_client = pb_client,
        **env_kwargs
    )
   
    if action_clip is not None:
        env = DeltaActionRestrainEnv(env, action_clip)
    if diagnal: 
        env = DiagonalActionEnv(env)
    
    return env

if __name__ =='__main__':
    task_config = dict(
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
    )

    env_config = dict(
        include_vision_obs = True,
        alive_height_range= [0.15, 0.35],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(forces= [40] * 12),
            simulate_timestep= 1./500,
            default_base_transform= [0, 0, 0.27, 0, 0, 0, 1],
        ),
        render_kwargs= dict(
            resolution= (64, 64),
            camera_name= "front",
            modal= "depth",
        ),
        prop_size = 33,
        history_info_num= dict(
            action = 3,
            image =4,
        ),
        step_rand_ratio = 1 ,
        nsubsteps = 10

    )

    env = build_locomotion_env(
        goal= False,
        rand = True,
        diagnal = True,
        terrain = 'thin',
        action_clip = [ 0.2,1,0.8,
                        0.2,1,0.8,
                        0.2,1,0.8,
                        0.2,1,0.8],
        task_kwargs = task_config,
        env_kwargs = env_config,
    )
    DEFAULT_ACTION = [0, 0.3, -0.5, 0, -0.3, 0.5]  
    for i in range(10000):
        if i % 2 == 0:
            flag = -1
        else:
            flag = 1
        obs, reward, done, info = env.step(np.array(DEFAULT_ACTION) * flag)
        
    

    

    