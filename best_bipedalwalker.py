import torch 
from classes.NEAT import *
species = torch.load('runs/bipedalwalker/2023-12-04 16:39:15.028425/species_1.pt')

import gymnasium as gym
env = gym.make("BipedalWalker-v3")

def fitness_f(network:NeuralNetwork, inputs, targets, print_fitness=False):
    global env
    #error = 0
    fitness = 0
    
    observation, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        input = {
            0:torch.tensor([1.0]),# bias
        }
        for i,o in enumerate(observation):
            input[i+1] = torch.tensor([o])
        
        actions = network.forward(input)
        actions = torch.tensor(actions)
        # sigmoid to tanh
        actions = 2*actions - 1
        # to python list
        action = actions.tolist()
        observation, reward, terminated, truncated, info = env.step(action)
        fitness += reward
   
    return fitness

from tqdm.auto import tqdm 


max_fitness = -10000
best_genotype = None
for genotype in tqdm(species[0].genotypes):

    network = NeuralNetwork(genotype)
    average_fitness = 0
    
    for _ in range(1):
        average_fitness += fitness_f(network, None, None).item()

    if average_fitness>max_fitness:
        max_fitness=average_fitness
        best_genotype = genotype

# env = gym.make("BipedalWalker-v3", render_mode='human')
# print(max_fitness)
# while True:
#     network = NeuralNetwork(genotype)
#     fitness = fitness_f(network, None, None)
from gymnasium.wrappers import RecordVideo

env = RecordVideo(gym.make("BipedalWalker-v3", render_mode='rgb_array')   , './video')     

# while True:
fitness = fitness_f(NeuralNetwork(best_genotype), None, None)
env.close()