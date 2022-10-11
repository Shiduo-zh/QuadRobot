import numpy as np
import pybullet as p
import random
from sympy import E
import torch
import argparse
import pickle
import os
from scibotpark.locomotion.envs.get_env import get_subprocvec_env, get_visual_env
from torchrl.algo import PPO
from torchrl.networks import LocoTransformer, LocoTransformerEncoder, MLPBase
from torchrl.collector.on_policy import VecOnPolicyCollector
from torchrl.policies.continuous_policy import GaussianContPolicyLocoTransformer
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import Logger
from scripts.run import get_static_param
from pybullet_utils import bullet_client
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    parser = argparse.ArgumentParser(description='viewer')
    parser.add_argument(
        '--log_dir',
        type = str,
        default = "trained",
        help = 'root directory to store config record'
    )
    parser.add_argument(
        '--model',
        type = str,
        default ='locotransformer',
        help = 'model type for policy'
    )

    parser.add_argument(
        '--model_name',
        type = str,
        default = 'model.pth',
        help = 'model state dict file for visulization'
    )
    parser.add_argument(
        '--env_name', 
        type=str, 
        default='A1MoveForward'
    )

    parser.add_argument(
        '--config',
        type = str,
        default='config file for basic setting'
    )

    parser.add_argument(
        '--id',
        type = str,
        default = 'thin-obs',
        help = 'experiment id for evaluation'
    )

    parser.add_argument(
        '--seed',
        type = int,
        default = 0
    )

    parser.add_argument(
        '--snap_check',
        type = str,
        default = 'best'
    )
        

    args = parser.parse_args()
    
    return args

def getPolicyCls(args):
    if args.model == 'locotransformer':
        return GaussianContPolicyLocoTransformer

def getEncoderCls(args):
    if args.model == 'locotransformer':
        return LocoTransformerEncoder

def view(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.log_dir = os.path.join(ROOT, args.log_dir)
    PARAM_PATH = "{}/{}/{}/{}/params.json".format(
        args.log_dir,
        args.id,
        args.env_name,
        args.seed
        )
    params = get_static_param(PARAM_PATH)
    params['env']['env_build']['use_gui'] = True
    env = get_visual_env(
        params['env_name'],
        params['env']
    )

    # load best normalizer state dict
    if hasattr(env, "_obs_normalizer"):
        NORM_PATH = "{}/{}/{}/{}/model/_obs_normalizer_{}.pkl".format(
            args.log_dir,
            args.id,
            params['env_name'],
            args.seed,
            args.snap_check
        )
        with open(NORM_PATH, 'rb') as f:
            env._obs_normalizer = pickle.load(f)
            print(env._obs_normalizer._mean)
            print(env._obs_normalizer._var)
   
    # load best policy model state dict
    image_channels = params['env']['env_build']['env_kwargs']['history_info_num']['image'] * 1 if \
                     params['env']['env_build']['env_kwargs']['render_kwargs']['modal'] == 'depth' else \
                     params['env']['env_build']['env_kwargs']['history_info_num']['image'] * 3  
    resolution = params['env']['env_build']['env_kwargs']['render_kwargs']['resolution'][0]
    
    EcoderCls = getEncoderCls(args)
    encoder = EcoderCls(
        in_channels = image_channels,
        state_input_dim = env.observation_space.shape[0] - image_channels * resolution ** 2,
        **params['encoder']
    )

    PolicyCls = getPolicyCls(args)
    policy = PolicyCls(
        encoder = encoder,
        state_input_shape = env.observation_space.shape[0] - image_channels * resolution **2,
        visual_input_shape = (image_channels, resolution, resolution),
        output_shape= env.action_space.shape[0],
        **params['net'],
        **params['policy']
    )
    PATH = "{}/{}/{}/{}/model/model_pf_{}.pth".format(
        args.log_dir,
        args.id,
        params['env_name'],
        args.seed,
        args.snap_check
    )
    policy.load_state_dict(
        torch.load(
            PATH,
            map_location='cuda:0'
        )
    )
    policy.eval()

    env.seed(args.seed)

    obs = env.reset()
    rewards = 0 
    steps = 0
    while True:
        obs = torch.Tensor(obs).unsqueeze(0)
        action = policy.eval_act(obs)
        print(action)

        obs, reward, done, info = env.step(action)
        steps += 1
        rewards += reward

        if done:
            print('epiode ended!')
            break
    
    print('view finished!')

if __name__ == '__main__':
    args = get_args()
    view(args)