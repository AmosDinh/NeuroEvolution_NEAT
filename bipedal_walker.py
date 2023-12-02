import random 
import torch 
import numpy as np
from classes.NEAT import *
random.seed(14)
torch.manual_seed(14)
np.random.seed(14)





n_networks = 150


# Fitness:
c1 = 1
c2 = 1
c3 = 0.4
distance_delta = 3


weight_magnitude = 2.5 # std of weight mutation
# Mutation
mutate_weight_prob = 0.8
mutate_weight_perturb = 0.8
mutate_weight_random = 1 - mutate_weight_perturb
mutate_add_node_prob = 0.02
mutate_add_link_prob_large_pop = 0.2
mutate_add_link_prob = 0.1
mutate_remove_link_prob = 0.03

offspring_without_crossover = 0.25
interspecies_mate_rate = 0.001

fitness_survival_rate = 0.2
interspecies_mate_rate = 0.001


node_gene_history = Node_Gene_History()
connection_gene_history = Connection_Gene_History()

genotypes = []


observation_space = 8 + 1
action_space = 4

for _ in range(n_networks):
    
    node_genes = []
    for i in range(observation_space):
        node_genes.append(Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=i))
    
    for i in range(observation_space,observation_space+action_space):
        node_genes.append(Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=1, initial_node_id=i))
    
    connection_genes = []
    for i in range(observation_space):
        for j in range(observation_space,observation_space+action_space):
            connection_genes.append(Connection_Gene(i, j, np.random.normal(), False, connection_gene_history))
    
    
    
    genotype = Genotype(
        node_genes, connection_genes, node_gene_history, connection_gene_history, 
        mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob, mutate_add_link_prob,mutate_remove_link_prob, weight_magnitude,
        c1, c2, c3)
    genotypes.append(genotype)






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


initial_species = Species(np.random.choice(genotypes), genotypes, distance_delta)


import os
import datetime 
now = datetime.datetime.now()
folder = os.path.join('runs', 'bipedalwalker', str(now))
    
evolved_species, solutions = evolve(
    features=None, 
    target=None, 
    fitness_function=fitness_f, 
    stop_at_fitness=1900, 
    n_generations=10000,
    species=[initial_species], 
    fitness_survival_rate=fitness_survival_rate, 
    interspecies_mate_rate=interspecies_mate_rate, 
    distance_delta=distance_delta,
    largest_species_linkadd_rate=mutate_add_link_prob_large_pop,
    eliminate_species_after_n_generations=20,
    run_folder=folder,
    elitism=True
    
)

print(solutions)