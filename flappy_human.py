import time

import pygame
import flappy_bird_gym
import numpy as np

def get_reward(done, location, score, new_score):
    if abs(location[1])>0.1 and not done:
        return -1
    if score < new_score and not done:
        return 5
    return -10 if done else 0


def play():
    # env = gym.make("flappy_bird_gym:FlappyBird-v0")
    env = flappy_bird_gym.make("FlappyBird-v0")
    clock = pygame.time.Clock()
    steps = 0

    obs = env.reset()

    score = 0
    while True:
        env.render()

        # Getting action:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if (event.type == pygame.KEYDOWN and
                    (event.key == pygame.K_SPACE or event.key == pygame.K_UP)):
                action = 1

        # Processing:
        next_obs, reward, done, info = env.step(action)
        new_score = info["score"]
        steps += reward
        #print(f"Obs: {obs}" + " Score: " + str(new_score) + " Reward: " + str(get_reward(done, obs, score, new_score)))
        print((np.round(obs*100), action))
        score = new_score
        clock.tick(30)
        obs = next_obs
        if done:
            env.render()
            time.sleep(0.6)
            break

    env.close()


if __name__ == "__main__":
    play()