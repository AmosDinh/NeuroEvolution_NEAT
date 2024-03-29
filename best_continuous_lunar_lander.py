import torch 
from classes.NEAT import *
species_id = 0
run = 237
# 2023-12-07 19:59:20.340927
# run 10, species 1 first uses thrusters
# run 12 species 0 first signs of steering
# run 15 0, strategy to stay in the air long: no fuel efficiency
# 1 explores dropping faster
# run 30 species 2: happy little jumper
# run = 80 # already good
# 96

# Folder: 2023-12-07 22:13:57.425933
# 18, 0 starts to steer
# 25, 2 strategy to rotate onto the goal
# 27, 2 minimal thrust, 0 faster
# 41, 2: almost like pid controller, 0 faster
# 90: you can see 0 more accurate, but 1 can have room for improvement, it falls faster: less fuel needed
# highest fitness: 125 species 1, 143 0

# folder: 2023-12-09 18:30:12.850515
# 237 species 0 is interesting, it does not look good but the "sidelands" have high fitness
# because it lands fast and uses side thrusters instead of main thruster
species = torch.load(f'runs/continuous_lunar_lander/2023-12-09 18:30:12.850515/species_{run}.pt')
fitnesses = torch.load(f'runs/continuous_lunar_lander/2023-12-09 18:30:12.850515/fitness_perspecies_{run}.pt')



import gymnasium as gym
import datetime
def lunar_fitness(genotype_and_env, inputs, targets):
    #error = 0
    genotype, env = genotype_and_env
    network = NeuralNetwork(genotype) 
    fitness = 0



    num_tries = 1
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
            actions = 2*actions - 1
            actions = actions.tolist()
            observation, reward, terminated, truncated, info = env.step(actions)
            fitness += reward
            
    fitness /= num_tries
    return fitness




genotypes = species[species_id].genotypes

best_genotype = genotypes[np.argmax(fitnesses[species_id])]
best_genotype.print_genotype()
for species in fitnesses:
    print(species, np.max(fitnesses[species]))

env = gym.make("LunarLander-v2", render_mode='human',continuous=True)       
env.reset() 
while True:
    fitness = lunar_fitness((best_genotype, env), None, None)
    print(fitness)
    
