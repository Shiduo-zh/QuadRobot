from scibotpark.locomotion.unitree import UnitreeForwardEnv
import pybullet as p
import numpy as np
from gym import spaces
import random
from pybullet_utils import bullet_client

class UnitreeLocomotionEnv(UnitreeForwardEnv):
    def __init__(self,
                *args,
                task=None,
                domain_randomization=dict(),
                prop_size,
                history_info_num=dict(),
                step_rand_ratio = 0.2,
                surrounding,
                **kwargs
                ):
        """
        param:
            history_info_num: includes the history images nums and history action nums 
            step_rand_ratio: the ratio to give the agent a disturbense 
        """
        self._task = task
        self._surroundings = surrounding
        
        super().__init__(*args,**kwargs)
        self.action_size = self.action_space.shape[0]
        self.prop_size = prop_size
        self.domain_rand_config = domain_randomization
        self.history_info_num = history_info_num
        self.step_num = 0
        self.step_rand_ratio = step_rand_ratio
        
        # properceptive initialization
        self.last_obs = np.zeros(self.prop_size*self.history_info_num['action'], dtype = np.float32)  # last up to n times prop take by agent
        self.last_action = np.zeros(self.action_size,dtype = np.float32 )  
        self.images = np.zeros((self.history_info_num['image'], self.render_kwargs['resolution'][0]**2),dtype = np.float32)

    def _build_surroundings(self):
        # load basic plane in super method
        super()._build_surroundings()
        # create other surrounings through surrounding
        # TODO :add surrounding method here
        self._surroundings.setup()
    
    def reset(self,*args,**kwargs):
        self.set_domain_rand()
        self.step_num=0
        self.last_obs = np.zeros(self.prop_size*self.history_info_num['action'])  # last up to n times prop take by agent
        self.last_action = np.zeros(self.action_size)  
        self.images=np.zeros((self.history_info_num['image'],self.render_kwargs['resolution']**2))
        return_=super().reset(*args,**kwargs)
        return return_    

    @property
    def observation_space(self):
        """
        return a dict with key 'vision' and 'proprioceptive'
        """
        base_observation_space = super().observation_space
        observation_space = spaces.Box(
            low= np.concatenate([
                base_observation_space["proprioceptive"].low[0:12],
                base_observation_space["proprioceptive"].low[30:36],
                base_observation_space["proprioceptive"].low[24:27],
                self.action_space.low,
            ]*self.history_info_num['action']),
            high= np.concatenate([
                base_observation_space["proprioceptive"].high[0:12],
                base_observation_space["proprioceptive"].high[30:36],
                base_observation_space["proprioceptive"].high[24:27],
                self.action_space.high,
            ]*self.history_info_num['action']),
        )

        if self.include_vision_obs:
            observation_space = spaces.Dict(dict(
                vision= base_observation_space["vision"],
                proprioceptive= observation_space,
            ))

        return observation_space
    
    def _get_obs(self):
        obs = super()._get_obs()
        delta_action=obs['proprioceptive'][0:self.action_size]-self.last_action
        cur_proprioceptive = np.concatenate([
                obs["proprioceptive"][0:12],
                obs["proprioceptive"][30:36],
                obs["proprioceptive"][24:27],
                delta_action,
            ])
        self.last_obs = np.concatenate([
            self.last_obs,
            cur_proprioceptive
        ])
        self.last_obs = self.last_obs[self.prop_size:]
        self.images = np.concatenate([
            self.images,
            obs['vision'].reshape(1, -1)
        ])
        self.images=self.images[1:]

        if self.include_vision_obs:
            obs=dict(
                proprioceptive = self.last_obs,
                vision = self.images
            )
        else:
            obs = self.last_obs
        return obs

    def step(self, action):
        self.set_domain_rand_step() # set domain radomization on each step
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        self.step_simulation_from_action(action)
        obs = self._get_obs()
        reward,reward_info, done = self._task.compute_reward(self)
        info=reward_info

        return obs, reward, done, info
    
    def set_domain_rand(self):
        """
        set different domain randomization in different episodes
        """
        #pd control domain randomization
        kp=random(self.domain_rand_config['pd control'][0][0],self.domain_rand_config[0][1])
        kd=random(self.domain_rand_config['pd control'][1][0],self.domain_rand_config[1][1])
        for joint_id in self.robot.valid_joint_ids:
            self.robot.pb_client.setJointMotorControl2(self.robot.body_id,
                                                joint_id,
                                                controlMode=p.POSITION_CONTROL,
                                                positionGain=kp,
                                                velocityGain=kd
                                                )
        #joint or link parameter settings
        inertial_coef=random.uniform(self.domain_rand_config['inertial'][0], self.domain_rand_config['inertial'][1])
        for link_id in range(1,self.robot.pb_client.getNumJoints(self.robot.body_id)):
            #leg joints settings
            if link_id in self.robot.valid_joint_ids:
                dynamic_infos = self.robot.pb_client.getDynamicsInfo(self.robot.body_id,link_id)
                mass = dynamic_infos[0]*random.uniform(self.domain_rand_config['mass'][0],self.domain_rand_config['mass'][1])
                inertial = [v*inertial_coef for v in dynamic_infos[2]]
                self.robot.pb_client.changeDynamics(self.robot.body_id,
                                                link_id,
                                                mass=mass,
                                                localInertiaDiagonal=inertial,
                                                )
        
            #feet joints settings
            else:
                dynamic_infos = self.robot.pb_client.getDynamicsInfo(self.robot.body_id,link_id)
                #random lateral foot friction 
                lateral_friction = random.uniform(self.domain_rand_config['lateral friction'][0],
                                                    self.domain_rand_config['lateral friction'][1])
                self.robot.pb_client.changeDynamics(self.robot.body_id,
                                                link_id,
                                                lateralFriction=lateral_friction)

        #motor strength setting
        forces_coef =[random.uniform(self.domain_rand_config['motor strength'][0], self.domain_rand_config['motor strength'][1]) \
                        for i in range(len(self.robot.valid_joint_ids))]
        forces = [force * force_coef for force, force_coef in zip(self.robot.pb_control_kwargs['forces'], forces_coef)]
        self.robot.pb_client.setJointMotorControlArray(
            self.robot.body_id,
            self.robot.valid_joint_ids,
            controlMode = self.robot.pb_control_mode,
            forces = forces,
        )
    
    def set_domain_rand_step(self):
        prob = random.random()
        if prob < self.domain_rand_config['external force ratio']:
            external_force=[
                random.randrange(self.domain_rand_config['random force'][0], self.domain_rand_config['random force'][1], 1),
                random.randrange(self.domain_rand_config['random force'][0], self.domain_rand_config['random force'][1], 1),
                random.randrange(self.domain_rand_config['random force'][0], 0.5*self.domain_rand_config['random force'][1], 1),
            ]
            position, _=self.robot.pb_client.getBasePositionAndOrientation(self.robot.body_id)
            self.pb_client.applyExternalForce(self.robot.body_id, -1, external_force, position,  p.WORLD_FRAME)
            # print('external! force is ',external_force)

    def compute_reward(self):
        return self._task.compute_reward(self)
    
if __name__ == "__main__":
    from scibotpark.locomotion.utils import get_default_randomization
    from scibotpark.locomotion.envs.task import forwardtask, goaltask
    DEFAULT_ACTION = [0, 0.8, -1.8] * 4

    domain_rand_setting = get_default_randomization()
    task = forwardtask.ForwardTask()
    
    env = UnitreeLocomotionEnv(
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
            pb_client= bullet_client.BulletClient(connection_mode= p.GUI),
            task = task,
            domain_redomization = domain_rand_setting,
            prop_size = 33,
            history_info_num= dict(
                action = 3,
                image =4,
            ),
            step_rand_ratio = 1 
    )

    while(1):
        env.step(np.array(DEFAULT_ACTION))



        

    

