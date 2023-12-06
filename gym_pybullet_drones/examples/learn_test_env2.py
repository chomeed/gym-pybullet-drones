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
from gym_pybullet_drones.envs.HoverAviaryEnv2 import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results/models/rgbkin'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('rgbkin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=False, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):
  
    train_env = make_vec_env(HoverAviary,
                                env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                n_envs=1,
                                seed=0
                                )
    eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = SAC('MultiInputPolicy',
                train_env,
                verbose=1)
       
    if os.path.isfile(output_folder+'/easy_task2/SAC.pkl'):
        print("CHECKPOINT FILE FOUND")
        path = output_folder+'/easy_task2/SAC.pkl'
        model = SAC.load(path, print_system_info=True)
        print("CHECKPOINT LOADED SUCCESFULLY")
    else:   
        print("[ERROR]: no model under the specified path", filename)
    
    #### Show (and record a video of) the model's performance ##

    # test_env = HoverAviary(gui=gui,
    #                         obs=DEFAULT_OBS,
    #                         act=DEFAULT_ACT,
    #                         record=record_video)
    test_env_nogui = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT)


    # logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
    #             num_drones=DEFAULT_AGENTS if multiagent else 1,
    #             output_folder=output_folder,
    #             colab=colab
    #             )

    # mean_reward, std_reward = evaluate_policy(model,
    #                                           test_env_nogui,
    #                                           n_eval_episodes=1
    #                                           )
    # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env_nogui.reset(seed=22, options={})
    start = time.time()
    totalReward = 0
    for i in range((test_env_nogui.EPISODE_LEN_SEC+500)*test_env_nogui.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        action = action.squeeze(0)
        obs, reward, terminated, truncated, info = test_env_nogui.step(action)
        # print(obs['obs_rgb'].shape, obs['obs_rgb'])
        totalReward += reward 
        # obs2 = obs.squeeze()
        act2 = action.squeeze()
        # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                pass
                # logger.log(drone=0,
                #     timestamp=i/test_env.CTRL_FREQ,
                #     state=np.hstack([obs2[0:3],
                #                         np.zeros(4),
                #                         obs2[3:15],
                #                         act2
                #                         ]),
                #     control=np.zeros(12)
                #     )
        # test_env.render() 
        # print(terminated)
        sync(i, start, test_env_nogui.CTRL_TIMESTEP)
        if terminated or truncated:
            print(totalReward)
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
    ARGS = parser.parse_args()

    run(**vars(ARGS))
