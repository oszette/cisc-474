import time

import pygame
import flappy_bird_gym


def play():
    # env = gym.make("flappy_bird_gym:FlappyBird-v0")
    env = flappy_bird_gym.make("FlappyBird-v0")

    clock = pygame.time.Clock()
    score = 0

    obs = env.reset()

    max_x = 0
    max_abs_y = 0

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
        obs, reward, done, info = env.step(action)

        score += reward
        print(f"Obs: {obs}")
        print(f"Score: {score}\n")

        if obs[0] > max_x:
            max_x = obs[0]
        if abs(obs[0]) > max_abs_y:
            max_abs_y = obs[1]

        clock.tick(15)

        if done:
            env.render()
            time.sleep(0.6)
            print(max_x)
            print(max_abs_y)
            break

    env.close()


if __name__ == "__main__":
    play()