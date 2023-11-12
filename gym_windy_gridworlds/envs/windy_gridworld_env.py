# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys


class WindyGridWorldEnv(gym.Env):
    """
    Creates the Windy GridWorld Environment
    """

    def __init__(
        self,
        grid_height=7,
        grid_width=10,
        wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        start_state=(3, 0),
        goal_state=(3, 7),
        reward=-1,
    ):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.wind = wind
        self.start_state = start_state
        self.goal_state = goal_state
        self.reward = reward
        self.action_space = spaces.Discrete(4)
        self.observation = start_state
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width))
        )
        self.actions = {"U": 0, "R": 1, "D": 2, "L": 3}  # up  # right  # down  # left

        # set up destinations for each action in each state
        self.action_destination = np.empty(
            (self.grid_height, self.grid_width), dtype=dict
        )
        for i in range(0, self.grid_height):
            for j in range(0, self.grid_width):
                destination = dict()
                destination[self.actions["U"]] = (max(i - 1 - self.wind[j], 0), j)
                destination[self.actions["D"]] = (
                    max(min(i + 1 - self.wind[j], self.grid_height - 1), 0),
                    j,
                )
                destination[self.actions["L"]] = (
                    max(i - self.wind[j], 0),
                    max(j - 1, 0),
                )
                destination[self.actions["R"]] = (
                    max(i - self.wind[j], 0),
                    min(j + 1, self.grid_width - 1),
                )
                self.action_destination[i, j] = destination
        self.nA = len(self.actions)

    def step(self, action):
        """
        Parameters
        ----------
        action : 0 = Up, 1 = Right, 2 = Down, 3 = Left

        Returns
        -------
        ob, reward, terminated, truncated, info : tuple
            ob (object) :
                 Agent current position in the grid.
            reward (float) :
                 Reward is -1 at every step.
            terminated (bool) :
                 True if the agent reaches the goal, False otherwise.
            truncated (bool) :
                 Used to indicate that the episode was ended before the agent reached the goal. In this environment, this will always be False.
            info (dict) :
                 Contains no additional information.
        """
        assert self.action_space.contains(action)
        self.observation = self.action_destination[self.observation][action]
        if self.observation == self.goal_state:
            return self.observation, -1, True, False, {}
        return self.observation, -1, False, False, {}

    def reset(self):
        """
        Resets the agent position back to the starting

        Returns:
            observation (object): the agent's current position in the grid
            info (dict): contains no additional information
        """
        self.observation = self.start_state
        return self.observation, {}

    def render(self, mode="human", close=False):
        """Renders the environment. Code borrowed and then modified
        from
        https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
        """
        outfile = sys.stdout
        nS = self.grid_height * self.grid_width
        shape = (self.grid_height, self.grid_width)

        outboard = ""
        for y in range(-1, self.grid_height + 1):
            outline = ""
            for x in range(-1, self.grid_width + 1):
                position = (y, x)
                if self.observation == position:
                    output = "X"
                elif position == self.goal_state:
                    output = "G"
                elif position == self.start_state:
                    output = "S"
                elif x in {-1, self.grid_width} or y in {-1, self.grid_height}:
                    output = "#"
                else:
                    output = " "

                if position[1] == shape[1]:
                    output += "\n"
                outline += output
            outboard += outline
        outboard += "\n"
        outfile.write(outboard)

    def seed(self, seed=None):
        pass
