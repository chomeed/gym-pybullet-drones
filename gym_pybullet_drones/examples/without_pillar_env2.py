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
import os
import time
import argparse
import gymnasium as gym
from stable_baselines3 import SAC

from gym_pybullet_drones.envs.HoverAviaryEnv2 import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False

DEFAULT_OBS = ObservationType('rgbkin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1

def run(gui=DEFAULT_GUI, plot=False, ):
  
 
  
    test_env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    print('[INFO] Action space:', test_env.action_space)
    print('[INFO] Observation space:', test_env.observation_space)

    model = SAC('MultiInputPolicy',
            test_env,
            verbose=1)

    obs, info = test_env.reset(seed=22, options={})
    start = time.time()
    totalReward = 0
    for i in range((test_env.EPISODE_LEN_SEC+50)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        action = action.squeeze(0)
        obs, reward, terminated, truncated, info = test_env.step(action)
        totalReward += reward 
        act2 = action.squeeze()

        sync(i, start, test_env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            print(totalReward)
            totalReward = 0
            obs, info = test_env.reset(seed=22, options={})
    test_env.close()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
