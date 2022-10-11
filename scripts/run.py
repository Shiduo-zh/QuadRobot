import numpy as np
import os
import random
import torch
import argparse
import json
from scibotpark.locomotion.envs.get_env import get_subprocvec_env
from torchrl.algo import PPO
from torchrl.networks import LocoTransformer, LocoTransformerEncoder, MLPBase
from torchrl.collector.on_policy import VecOnPolicyCollector
from torchrl.policies.continuous_policy import GaussianContPolicyLocoTransformer
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import Logger

# ROOT = 'D:\Program Files\GitHub-Collection\GitHub\QuadRobot'
ROOT = '/home/zhangsd/project/QuadRobot'
def get_static_param(file_name):
    with open(file_name) as f:
        params = json.load(f)
        params['env']['env_build']['env_kwargs']['render_kwargs']['resolution'] = tuple(params['env']['env_build']['env_kwargs']['render_kwargs']['resolution'])
    return params

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')

    parser.add_argument('--vec_env_nums', type=int, default=1,
                        help='vec env nums')

    parser.add_argument('--proc_nums', type=int, default=1,
                        help='vec env nums')

    parser.add_argument('--eval_worker_nums', type=int, default=2,
                        help='eval worker nums')

    parser.add_argument("--config", type=str,   default='thin_heightfield.json',
                        help="config file",)

    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')

    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='overwrite previous experiments')

    parser.add_argument("--device", type=int, default=0,
                        help="gpu secification",)

    # tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard",)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def run_experiment(args, params):
    

    # train and evaluation environment setting
    env = get_subprocvec_env(
        params["env_name"],
        params["env"],
        args.vec_env_nums,
        args.proc_nums
    )

    eval_env = get_subprocvec_env(
        params["env_name"],
        params["env"],
        max(2, args.vec_env_nums),
        max(2, args.proc_nums)
    )

    if hasattr(env, "_obs_normalizer"):
        eval_env._obs_normalizer = env._obs_normalizer
    
    # basic configuration on seed and cuda device
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = False
    
    # logger , buffer and divice initailization
    device = torch.device(
        "cuda:{}".format(args.device)
    )

    experiment_name = os.path.split(
        os.path.splitext(args.config)[0])[-1] if args.id is None else args.id
    
    logger = Logger(
        experiment_name,
        params["env_name"],
        args.seed,
        params,
        args.log_dir,
        args.overwrite
    )
    params['general_setting']['env'] = env
    
    replay_buffer = OnPolicyReplayBuffer(
        env_nums = args.vec_env_nums,
        max_replay_buffer_size = int(params['replay_buffer']['size']),
        time_limit_filter = params['replay_buffer']['time_limit_filter']
    )

    params['general_setting']['replay_buffer'] = replay_buffer
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device
    params['net']['base_type'] = MLPBase

    # actor policy and critic model initilization
    image_channels = params['env']['env_build']['env_kwargs']['history_info_num']['image'] * 1 if \
                     params['env']['env_build']['env_kwargs']['render_kwargs']['modal'] == 'depth' else \
                     params['env']['env_build']['env_kwargs']['history_info_num']['image'] * 3  
    resolution = params['env']['env_build']['env_kwargs']['render_kwargs']['resolution'][0]
    encoder = LocoTransformerEncoder(
        in_channels = image_channels,
        state_input_dim = env.observation_space.shape[0] - image_channels*resolution**2,
        **params['encoder']
    )
    print(env.observation_space.shape[0] - image_channels*resolution**2)
    policy = GaussianContPolicyLocoTransformer(
        encoder = encoder,
        state_input_shape = env.observation_space.shape[0] - image_channels*resolution**2,
        visual_input_shape = (image_channels, resolution, resolution),
        output_shape = env.action_space.shape[0],
        **params['net'],
        **params['policy']
    )

    critic = LocoTransformer(
        encoder = encoder,
        state_input_shape = env.observation_space.shape[0] - image_channels*resolution**2,
        visual_input_shape = (image_channels, resolution, resolution),
        output_shape = 1,
        **params['net']
    )
    # print(policy)


    # collector initialization
    collector = VecOnPolicyCollector(
        vf = critic,
        env = env,
        eval_env = eval_env,
        pf = policy,
        replay_buffer = replay_buffer,
        device = device,
        train_render = False,
        **params['collector']
    )

    params['general_setting']['collector'] = collector
    params['general_setting']['save_dir'] = os.path.join(
        logger.work_dir, "model"
    )
    agent = PPO(
        pf = policy,
        vf = critic,
        **params["ppo"],
        **params["general_setting"]
    )
    print('experiment begins...')
    agent.train()

if __name__ == '__main__':
    from scibotpark.locomotion.envs.get_env import *
    args = get_args()
    params = get_static_param(os.path.join(ROOT, 'config', args.config))
    # env = get_single_env(
    #     params["env_name"],
    #     params["env"],
    # )
    # obs = env.reset()
    # print(env.observation_space)

    run_experiment(args, params)


"""
example:
python scripts/run.py  --seed 0 --log_dir logs --id thin-obs  --env_name A1MoveForward --proc_nums 1 --vec_env_nums 1 --config this-obs.json
"""