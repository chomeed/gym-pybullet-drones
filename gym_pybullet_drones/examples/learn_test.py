"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the SAC algorithm.

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
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
DEFAULT_ENV_SIZE = 'large'
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'our_model'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('rgbkin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

def run(env_size=DEFAULT_ENV_SIZE, multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=False, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):
    print(env_size)

    test_env_nogui = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, env_size=env_size)

    #### Train the model #######################################
    model = SAC('MultiInputPolicy',
                test_env_nogui,
                verbose=1)

    if os.path.isfile(output_folder+'/93.pkl'): # or use '/92.pkl'
        print("CHECKPOINT FILE FOUND")
        path = output_folder+'/93.pkl'
        model = SAC.load(path, print_system_info=True)
        print("CHECKPOINT LOADED SUCCESFULLY")
    else:   
        print("[ERROR]: no model under the specified path")
    
    #### Show (and record a video of) the model's performance ##

    obs, info = test_env_nogui.reset(seed=22, options={})
    start = time.time()
    totalReward = 0
    for i in range((test_env_nogui.EPISODE_LEN_SEC+500)*test_env_nogui.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        action = action.squeeze(0)
        obs, reward, terminated, truncated, info = test_env_nogui.step(action)
        totalReward += reward 
        act2 = action.squeeze()

        sync(i, start, test_env_nogui.CTRL_TIMESTEP)
        if terminated or truncated:
            print('-'*99)
            print(totalReward)
            print('-'*99)
            totalReward = 0
            obs, info = test_env_nogui.reset(seed=22, options={})
    test_env_nogui.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--env_size',           default=DEFAULT_ENV_SIZE,      type=str)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
