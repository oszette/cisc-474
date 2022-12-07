from Flappy import Flappy
import numpy as np
import matplotlib.pyplot as plt


max_x = 170
max_y = 70

step_x = 1
step_y = 1

bird = Flappy(max_x, max_y, step_x, step_y, epsilon=0.1, alpha=0.75, gamma=0.95, nbr_episodes=5000, test_agent=True, test_every_eps=500)
bird.sarsa_learning()
plt.plot(bird.steps_and_scores)
plt.show()