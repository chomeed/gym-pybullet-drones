import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

import random

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        targetX = random.uniform(-2, 2)
        targetY = random.uniform(-2, 2)
        self.TARGET_POS = np.array([targetX,targetY,1])
        self.EPISODE_LEN_SEC = 8
        self.prevDisplacement = np.linalg.norm(self.TARGET_POS - self.INIT_XYZS)
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+'goal_position.urdf'), self.TARGET_POS,
                                                    p.getQuaternionFromEuler([0,0,0]),
                                                    useFixedBase=True,   # Doesn't move
                                                    #flags = p.URDF_USE_INERTIA_FROM_FILE,
                                                    physicsClientId=self.CLIENT
                                                )

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = 0 

        # 거리 - 이전 좌표와의 거리와 비교 
        # self.distPrev
        currentPosition = state[0:3] 
        currentDisplacement = np.linalg.norm(self.TARGET_POS - currentPosition)
        
        if currentDisplacement < self.prevDisplacement: # closer 
            ret += 1 
        else: 
            ret += -1 
            
        # 정확한 위치 - 추가 보상 
        if currentDisplacement < 1:
            ret += 1 - np.linalg.norm(self.TARGET_POS-state[0:3])

        # 시간 
        ret -= 0.02 

        # 진동 
        # 기울어짐 
        # 경로의 급격한 변화
        # pillar 충돌 방지  
        # 초록색 박스에서 속도 줄이기 

        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        import math
        state = self._getDroneStateVector(0)
        roll, pitch = state[7:9]
        x, y, z = state[:3]

        # if roll > math.pi/2 or roll < -math.pi/2 or pitch > math.pi/2 or pitch < -math.pi/2:
        #     return True 
        if abs(roll) > 2.967 or abs(pitch) > 2.967: # 170도 이상 회전하면 terminate
            return True 
        if abs(x) > 3 or abs(y) > 3: 
            return True 
        elif z < 0.2 or z > 4: 
            return True 
        else:
            return False    

        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
        #     return True
        
        # Terminates when time is over
        # if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
        #     return True
        # else:
        #     return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years


    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        targetX = random.uniform(1, 2)
        targetY = random.uniform(-1, 1)
        self.TARGET_POS = np.array([targetX,targetY,1])
        
        return initial_obs, initial_info
