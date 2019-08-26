#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:44:23 2019

@author: tawehbeysolow
"""

import cv2, gym, numpy as np
from retro_contest.local import make
from retro import make as make_retro
from baselines.common.atari_wrappers import FrameStack

cv2.ocl.setUseOpenCL(False)

class PreprocessFrame(gym.ObservationWrapper):
    """
    Grayscaling image from three dimensional RGB pixelated images
    - Set frame to gray
    - Resize the frame to 96x96x1
    """
    def __init__(self, environment, width, height):
        gym.ObservationWrapper.__init__(self, environment)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=255,
                                                shape=(self.height, self.width, 1), 
                                                dtype=np.uint8)

    def observation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = image[:, :, None]
        return image


class ActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []

        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True
            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            _actions = np.array([False] * len(buttons))
            for button in action:
                _actions[buttons.index(button)] = True
            self._actions.append(_actions)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): 
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):

        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, environment):
        super(AllowBacktracking, self).__init__(environment)
        self.curent_reward = 0
        self.max_reward = 0

    def reset(self, **kwargs):
        self.current_reward = 0
        self.max_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.environment.step(action)
        self.current_reward += reward
        reward = max(0, self.current_reward - self.max_reward)
        self.max_reward = max(self.max_reward, self.current_reward)
        return observation, reward, done, info

def wrap_environment(environment, n_frames=4):
    environment = ActionsDiscretizer(environment)
    environment = RewardScaler(environment)
    environment = PreprocessFrame(environment)
    environment = FrameStack(environment, n_frames)
    environment = AllowBacktracking(environment)
    return environment

def create_new_environment(environment_index, n_frames=4):
    """
    Create an environment with some standard wrappers.
    """

    dictionary = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}]
    
    print(dictionary[environment_index]['game'])
    print(dictionary[environment_index]['state'])
    
    environment = make(game=dictionary[environment_index]['game'], 
                       state=dictionary[environment_index]['state'],
                       bk2dir="./records")

    environment = wrap_environment(environment=environment,
                                   n_frames=n_frames)
    
    return environment


def make_test_level_Green():
    return make_test()


def make_test(n_frames=4):
    """
    Create an environment with some standard wrappers.
    """

    environment = make_retro(game='SonicTheHedgehog-Genesis', 
                             state='GreenHillZone.Act2', 
                             record="./records")

    environment = wrap_environment(environment=environment,
                                   n_frames=n_frames)

    return environment

