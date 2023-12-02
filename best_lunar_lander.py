import torch 
from classes.NEAT import *
species = torch.load('runs/lunar/2023-12-01 22:36:50.722894/species_459.pt')

import gymnasium as gym
env = gym.make("LunarLander-v2")

def lunar_fitness(network:NeuralNetwork, inputs, targets, print_fitness=False, env=None):
    #error = 0
    fitness = 1000
    
    observation, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        input = {
            0:torch.tensor([1.0]),# bias
            1:torch.tensor([observation[0]]),
            2:torch.tensor([observation[1]]),
            3:torch.tensor([observation[2]]),
            4:torch.tensor([observation[3]]), 
            5:torch.tensor([observation[4]]),
            6:torch.tensor([observation[5]]),
            7:torch.tensor([observation[6]]),
            8:torch.tensor([observation[7]]), 
        }
        actions = network.forward(input)
        actions = torch.tensor(actions)
        action = torch.argmax(actions).item()
        observation, reward, terminated, truncated, info = env.step(action)
        fitness += reward
   
    return fitness

from tqdm.auto import tqdm 

max_fitness = 0
best_genotype = None
for genotype in tqdm(species[0].genotypes):

    network = NeuralNetwork(genotype)
    average_fitness = 0
    
    for _ in range(3):
        average_fitness += lunar_fitness(network, None, None).item()

    if average_fitness>max_fitness:
        max_fitness=average_fitness
        best_genotype = genotype

print(max_fitness/3)
env = gym.make("LunarLander-v2", render_mode='human')        
while True:
    network = NeuralNetwork(genotype)
    fitness = lunar_fitness(network, None, None)
