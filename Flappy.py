#import time
#import pygame
import flappy_bird_gym
import numpy as np
import itertools

class Flappy:

    def __init__(self, max_x, max_y, step_x, step_y, epsilon, alpha, gamma):
        self.state_space_x = self.linspace(0, max_x, step_x) #vector of discritized state space for x
        self.state_space_y = self.linspace(-max_y, max_y, step_y) #above but for y
        self.qtable = np.zeros((len(self.state_space_x)*len(self.state_space_y), 2)) #qtable
        self.discrete_state_space = list(itertools.product(self.state_space_x, self.state_space_y))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.default_reward = 0
        self.death_reward = -1
        self.actions = [0, 1]
        self.env = flappy_bird_gym.make("FlappyBird-v0")

    #help function
    def linspace(self, start, stop, step_size):
        return np.around(np.linspace(start, stop, int((stop - start) / step_size + 1)), 2)

    #returns the state correspoding to the location (the position index of location in self.discrete_state_space)
    def get_state_from_location(self, location):
        return self.discrete_state_space.index(location) 

    #returns the q_value from the q_table for the pair (location, action)
    def get_qvalue(self, location, action):
        return self.qtable[self.get_state_from_location(location), action]

    #sets the qvalue for the pair (location, action) to the specified value
    def set_qvalue(self, location, action, value):
        self.qtable[self.get_state_from_location(location), action] = value

    #returns the epsilon greedy action based on the current location
    def eps_greedy_action(self, location):
        if self.epsilon > np.random.uniform():
            return self.actions[np.random.randint(len(self.actions))]
        return self.actions[np.argmax(self.qtable[self.get_state_from_location(location)])]
    
    #returns reward based on whether or not the bird has died or not (boolean value)
    def get_reward(self, died):
        return self.death_reward if died else self.default_reward
    
    #sarsa learning
    def sarsa_learning(self):
        #resets the environment and returns the initial location (discretized to fit the state-space)
        location = self.get_discrete_value(self.env.reset())