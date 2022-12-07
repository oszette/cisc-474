import time
import pygame
import flappy_bird_gym
import numpy as np
import itertools

class Flappy:

    #nbr of decimals of x and y in the rounding needs to be changed if the step_x and step_y are decreased to 0.01
    def __init__(self, max_x, max_y, step_x, step_y, epsilon, alpha, gamma, nbr_episodes, test_agent, test_every_eps):
        self.state_space_x = self.linspace(0, max_x, step_x) #vector of discritized state space for x
        self.state_space_y = self.linspace(-max_y, max_y, step_y) #above but for y
        self.qtable = np.zeros((len(self.state_space_x)*len(self.state_space_y), 2)) #qtable
        self.discrete_state_space = list(itertools.product(self.state_space_x, self.state_space_y))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.default_reward = 0
        self.bad_region_reward = -1
        self.score_reward = 10
        self.death_reward = -50
        self.actions = [0, 1]
        self.delay = 1
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.nbr_episodes = nbr_episodes
        self.steps_and_scores = np.zeros((nbr_episodes, 2))
        self.test_agent = test_agent
        self.test_every_eps = test_every_eps

    #help function
    def linspace(self, start, stop, step_size):
        return np.linspace(start, stop, int((stop - start) / step_size + 1))

    #returns the state correspoding to the location (the position index of location in self.discrete_state_space)
    def get_state_from_location(self, location):
        x = np.round(location[0]*100)
        y = np.round(location[1]*100)
        return self.discrete_state_space.index((x, y)) 

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

    #returns greedy action (only used in the agent_play function)
    def greedy_action(self, location):
        return self.actions[np.argmax(self.qtable[self.get_state_from_location(location)])]

    #returns reward based on whether or not the bird has died or not (boolean value)
    def get_reward(self, done, location, score, new_score):
        if done:
            return self.death_reward
        if score < new_score:
            return self.score_reward
        return self.default_reward
    
    def step(self, action):
        self.env.step(action)
        for i in range(self.delay-1):
            self.env.step(0)
        return self.env.step(0)
    
    #sarsa learning
    def sample_sarsa_episode(self):
        #resets the environment and returns the initial location (discretized to fit the state-space)
        location = self.env.reset()
        action = self.eps_greedy_action(location)
        steps = 0
        score = 0
        while True:
            next_location, reward, done, info = self.env.step(action)
            new_score = info["score"]
            next_action = self.eps_greedy_action(next_location)
            qvalue = (1-self.alpha) * self.get_qvalue(location, action) + self.alpha * (self.get_reward(done, next_location, score, new_score) + self.gamma*self.get_qvalue(next_location, next_action) - self.get_qvalue(location, action))
            self.set_qvalue(location, action, qvalue)
            location = next_location
            action = next_action
            score = new_score
            #self.env.render()
            if done:
                break
            steps += reward
        return (steps, score)

    def sarsa_learning(self):
        for eps_idx in range(self.nbr_episodes):
            if eps_idx % 500 == 0 and eps_idx > 0 == 0:
                self.epsilon = self.epsilon/10
            self.steps_and_scores[eps_idx] = self.sample_sarsa_episode()
            print("Episode " + str(eps_idx) + ": Steps/Score " + str((int(self.steps_and_scores[eps_idx][0]))) + "/" + str((int(self.steps_and_scores[eps_idx][1]))) + ", eps " + str(self.epsilon))
            if self.test_agent and eps_idx % self.test_every_eps == 0  and eps_idx > 0:
                #np.savetxt(str(eps_idx)+".csv", self.qtable, fmt="%f", delimiter=",")
                self.agent_play()
    
    def sample_q_episode(self):
        location = self.env.reset()
        steps = 0
        score = 0
        while True:
            action = self.eps_greedy_action(location)
            next_location, reward, done, info = self.env.step(action)
            new_score = info["score"]
            qvalue = (1-self.alpha) * self.get_qvalue(location, action) + self.alpha * (self.get_reward(done, next_location, score, new_score) + self.gamma*np.max(self.qtable[self.get_state_from_location(next_location)]) - self.get_qvalue(location, action))
            self.set_qvalue(location, action, qvalue)
            location = next_location
            score = new_score
            #self.env.render()
            if done:
                break
            steps += reward
        return (steps, score)
    
    def q_learning(self):
        for eps_idx in range(self.nbr_episodes):
            if eps_idx % 1000 == 0 and eps_idx > 0 == 0:
                self.epsilon = self.epsilon/10
            self.steps_and_scores[eps_idx] = self.sample_q_episode()
            print("Episode " + str(eps_idx) + ": Steps/Score " + str((int(self.steps_and_scores[eps_idx][0]))) + "/" + str((int(self.steps_and_scores[eps_idx][1]))) + ", eps " + str(self.epsilon))
            if self.test_agent and eps_idx % self.test_every_eps == 0 and eps_idx > 0:
                #np.savetxt(str(eps_idx)+".csv", self.qtable, fmt="%f", delimiter=",")
                self.agent_play()

    
    #renders the agent playing with the current q-table
    def agent_play(self):
        obs = self.env.reset()
        score = 0
        while True:
            self.env.render()
            action = self.greedy_action(obs)
            obs, reward, done, _ = self.env.step(action)
            print(f"Obs: {obs}\n"
              f"Score: {score}\n")

            time.sleep(1 / 30)

            if done:
                self.env.close()
                break
            score += reward

