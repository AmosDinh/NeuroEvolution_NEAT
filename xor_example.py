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
distance_delta = 6


weight_magnitude = 2.5 # std of weight mutation
# Mutation
mutate_weight_prob = 0.8
mutate_weight_perturb = 0.8
mutate_weight_random = 1 - mutate_weight_perturb
mutate_add_node_prob = 0.02
mutate_add_link_prob_large_pop = 0.08
mutate_add_link_prob = 0.02

offspring_without_crossover = 0.25
interspecies_mate_rate = 0.001

fitness_survival_rate = 0.2
interspecies_mate_rate = 0.001


node_gene_history = Node_Gene_History()
connection_gene_history = Connection_Gene_History()

genotypes = []


for _ in range(n_networks):
    node_genes = [
        Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=0), 
        Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=1),
        Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=2),
        Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=1, initial_node_id=3)
    ]
    
    connection_genes = [
        Connection_Gene(0, 3, np.random.normal(), False, connection_gene_history), # bias
        Connection_Gene(1, 3, np.random.normal(), False, connection_gene_history), # input 1 
        Connection_Gene(2, 3, np.random.normal(), False, connection_gene_history), # input 2
    ]
    
    genotype = Genotype(
        node_genes, connection_genes, node_gene_history, connection_gene_history, 
        mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob, mutate_add_link_prob, weight_magnitude,
        c1, c2, c3)
    genotypes.append(genotype)

# %%
# xor
inputs = [
    {0:torch.tensor([1.0]),1:torch.tensor([0.0]),2:torch.tensor([0.0])},
    {0:torch.tensor([1.0]),1:torch.tensor([1.0]),2:torch.tensor([0.0])},
    {0:torch.tensor([1.0]),1:torch.tensor([0.0]),2:torch.tensor([1.0])},
    {0:torch.tensor([1.0]),1:torch.tensor([1.0]),2:torch.tensor([1.0])}
    # xor:
    # bias 1, 00, 01, 10, 11
]
targets = [
    torch.tensor([0.0]),
    torch.tensor([1.0]),
    torch.tensor([1.0]),
    torch.tensor([0.0])
]

# %%
len(genotypes)

# %%

initial_species = Species(np.random.choice(genotypes), genotypes, distance_delta)

evolved_species, solutions = evolve(
    features=inputs, 
    target=targets, 
    fitness_function=xor_fitness, 
    stop_at_fitness=3.85, 
    n_generations=1000,
    species=[initial_species], 
    fitness_survival_rate=fitness_survival_rate, 
    interspecies_mate_rate=interspecies_mate_rate, 
    distance_delta=distance_delta,
    largest_species_linkadd_rate=mutate_add_link_prob_large_pop,
    eliminate_species_after_n_generations=20
    
)

print(solutions)