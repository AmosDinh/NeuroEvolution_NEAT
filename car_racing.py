import random 
import torch 
import numpy as np
from classes.NEAT import *
random.seed(14)
torch.manual_seed(14)
np.random.seed(14)





n_networks = 10

 
# Fitness:
c1 = 1
c2 = 1
c3 = 0.5
distance_delta = 40


weight_magnitude = 1.5 # std of weight mutation
# Mutation
mutate_weight_prob = 0.8
mutate_weight_perturb = 0.8
mutate_weight_random = 1 - mutate_weight_perturb
mutate_add_node_prob = 0.3
mutate_remove_node_prob = 0.3
mutate_add_link_prob_large_pop = 0.3
mutate_add_link_prob = 0.3
mutate_remove_link_prob = 0.3

#offspring_without_crossover = 0.05
interspecies_mate_rate = 0.001

fitness_survival_rate = 0.2
interspecies_mate_rate = 0.001


node_gene_history = Node_Gene_History()
connection_gene_history = Connection_Gene_History()

genotypes = []


for _ in range(n_networks):
    
    node_genes = []
    for i in range(96*96*1+1):
        node_genes.append(Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=i))
    
    for i in range(96*96*1+1,96*96*1+1+3):
        node_genes.append(Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=1, initial_node_id=i))
    
    connection_genes = []
    for i in range(96*96*1+1):
       for j in range(96*96*1+1,96*96*1+1+3):
           connection_genes.append(Connection_Gene(i, j, np.random.normal(), False, connection_gene_history))
     
    # initially only connect bias to output
    # connection_genes.append(Connection_Gene(0, 9, np.random.normal(), False, connection_gene_history))    
    # connection_genes.append(Connection_Gene(0, 10, np.random.normal(), False, connection_gene_history))    
    
    
    genotype = Genotype(
        node_genes, connection_genes, node_gene_history, connection_gene_history, 
        mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob,mutate_remove_node_prob, mutate_add_link_prob,mutate_remove_link_prob, weight_magnitude,
        c1, c2, c3)
    genotypes.append(genotype)






import gymnasium as gym
#env = gym.make("LunarLander-v2")





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
            x = torch.tensor(observation).view(-1).to(torch.float32)
            input = {i:x[i].unsqueeze(0) for i in range(1,96*96*1+1)}
            input[0] = torch.tensor([1.0])# bias
           
            actions = network.forward(input)
            actions = torch.tensor(actions)*2 -1
            actions = actions.tolist()
            observation, reward, terminated, truncated, info = env.step(actions)
            fitness += reward
            
            
    fitness /= num_tries
    env.close()
    return fitness


initial_species = Species(np.random.choice(genotypes), genotypes, distance_delta)


import os
import datetime 
now = datetime.datetime.now()
folder = os.path.join('runs', 'car_racing', str(now))
    
evolved_species, solutions = evolve(
    features=None, 
    target=None, 
    fitness_function=lunar_fitness, 
    stop_at_fitness=1000, 
    n_generations=10000,
    species=[initial_species], 
    fitness_survival_rate=fitness_survival_rate, 
    interspecies_mate_rate=interspecies_mate_rate, 
    distance_delta=distance_delta,
    largest_species_linkadd_rate=mutate_add_link_prob_large_pop,
    eliminate_species_after_n_generations=20,
    run_folder=folder,
    n_workers=1,
    gymnasium_env=[gym.make( "CarRacing-v2",continuous=True) for _ in range(n_networks)],
    elitism=True
    
)

print(solutions)