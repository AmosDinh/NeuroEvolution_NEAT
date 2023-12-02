import gymnasium as gym
env = gym.make("LunarLander-v2") # render_mode="human"
observation, info = env.reset()
fitness = 0
for _ in range(10000):
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    action =  1
    #print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    fitness += reward
    if terminated or truncated:
        observation, info = env.reset()
        print(fitness)
        fitness = 0

env.close()