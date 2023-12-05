"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor



import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('rgbkin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

DEFAULT_STEPS = 300000

def run(checkpoint_dir=None, steps=DEFAULT_STEPS, multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=False, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, wb_run=None, **kwargs):

    print("Steps", steps)
    if wb_run == None: 
        print("Wandb Key is not given")
        return

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    if not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=10,
                                 seed=0
                                 )
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        print("Single Agent Only")
        return

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = SAC('MultiInputPolicy',
                train_env,
                # policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]),
                # tensorboard_log=filename+'/tb/',
                tensorboard_log=output_folder + f"/runs/{wb_run.id}",
                verbose=1)
    # print(os.listdir('/root/models'))
    if checkpoint_dir is not None:
        print(os.listdir(checkpoint_dir))
        if os.path.isfile(checkpoint_dir+'/hover_without_green.pkl'):
            print("시작해보자222")
            path = checkpoint_dir+'/hover_without_green.pkl'
            model = SAC.load(path, env=train_env, print_system_info=True, tensorboard_log=output_folder + f"/runs/{wb_run.id}",
                    verbose=1)
            # model = model.load(path, print_system_info=True)
            print("끝")
        else: 
            print("not found")
    # path = '/models/hover_without_green.pkl'
    # model = SAC.load(path, print_system_info=True)

    model.learn(total_timesteps=int(steps) if local else 6*int(1e3), # shorter training in GitHub Actions pytest
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=output_folder + f"/models/{wb_run.id}",
                    verbose=2
                ),
                log_interval=25)

    #### Save the model ########################################
    model.save(output_folder+'/models/success_model.pkl')

    #### Print training progression ############################
    # with np.load(filename+'/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if os.path.isfile(output_folder+'/models/success_model.pkl'):
        path = output_folder+'/models/success_model.pkl'
        model = SAC.load(path)
        print(path, " loaded successfully.")
    elif os.path.isfile(filename+'/best_model.pkl'):
        path = filename+'/best_model.pkl'
        model = SAC.load(path)
    else:
        print("[ERROR]: no model under the specified path", filename) 
    
    #### Show (and record a video of) the model's performance ##
    print("TEST ENV RECORD VIDEO:", record_video)
    test_env = HoverAviary(gui=gui,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            record=record_video)
    test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--wandb_key')
    parser.add_argument('--checkpoint_dir',      default=None)
    parser.add_argument('--steps',              default=DEFAULT_STEPS)
    ARGS = parser.parse_args()

    wandb.login(key=ARGS.wandb_key)

    wb_run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    run(**vars(ARGS), wb_run=wb_run)
