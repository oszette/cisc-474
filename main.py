from Flappy import Flappy
import numpy as np

max_x = 1.7
max_y = 0.7

step_x = 0.01
step_y = 0.01

bird = Flappy(max_x, max_y, step_x, step_y, epsilon=0.01, alpha=0.5, gamma=0.5, nbr_episodes=1000)
print(bird.sarsa_learning())