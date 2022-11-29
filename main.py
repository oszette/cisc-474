from Flappy import Flappy
import numpy as np

max_x = 2
max_y = 1

step_x = 0.01
step_y = 0.01

bird = Flappy(max_x, max_y, step_x, step_y, epsilon=0.1, alpha=0.5, gamma=0.5, nbr_episodes=100)
print(bird.sarsa_learning())