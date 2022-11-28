from Flappy import Flappy
import numpy as np

max_x = 2
max_y = 1

step_x = 1
step_y = 1

bird = Flappy(max_x, max_y, step_x, step_y, epsilon=0.1, alpha=0.5, gamma=0.5)
bird.set_qvalue((0,-1), 1, 1)
print(bird.discrete_state_space)
print(bird.qtable)
print(bird.eps_greedy_action((0,-1)))
