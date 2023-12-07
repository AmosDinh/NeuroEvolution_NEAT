import torch 
from classes.NEAT import *
#174
#386
species = torch.load('runs/lunar/2023-12-03 15:15:59.070545/species_18.pt')

import gymnasium as gym
import datetime
def lunar_fitness(genotype_and_env, inputs, targets):
    #error = 0
    genotype, env = genotype_and_env
    network = NeuralNetwork(genotype) 
    fitness = 0
 
    
    
    num_tries = 4
    for _ in range(num_tries):
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
            
    fitness /= num_tries
    return fitness




from tqdm.auto import tqdm 
env = gym.make("LunarLander-v2")
max_fitness = -np.inf
best_genotype = None

n_workers = 8 
gymnasium_env = [gym.make("LunarLander-v2") for _ in range(150)]
genotypes = species[0].genotypes
with Pool(n_workers) as p:
    fitnesses = p.map(partial(lunar_fitness, inputs=None, targets=None), zip(genotypes, gymnasium_env[:len(genotypes)]))

best_genotype = genotypes[np.argmax(fitnesses)]

print(np.max(fitnesses))
env = gym.make("LunarLander-v2", render_mode='human')       
env.reset() 
while True:
    fitness = lunar_fitness((best_genotype, env), None, None)
