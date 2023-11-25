# %%
import numpy as np 

# %% [markdown]
# # NEAT
# 1. Classes and Functions
# - 1.1. Neural Network (Genotype)->Phenotype, input, output dim, contains mutation: 
# - 1.2. Genotype: A->B: connection gene, A:Node gene, is_disabled, weight, keep track of gene history
# - 1.3. Crossover (Genotype1, Genotype2)->Genotype12
# - 1.4. Species, represented by random member
# - 1.5. Speciation (List of Species)-> List of Species
# - 1.6. Fitness Calculation (Species)
# - 1.7.

# %%
# Evolver:
n_networks = 150

# Fitness:
c1 = 1.0
c2 = 1.0
c3 = 0.4
dt = 3 

# Mutation
mutate_weight_prob = 0.8
mutate_weight_perturb = 0.9
mutate_weight_random = 1 - mutate_weight_perturb
mutate_add_node_prob = 0.03
mutate_add_node_prob_large_pop = 0.3
mutate_add_link_prob = 0.05

offspring_without_crossover = 0.25
interspecies_mate_rate = 0.001

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-4.9*x))


# %%

class History:
    def __init__(self):
        self.last_node_id = 0
        self.last_connection_id = 0
        self.node_innovations = {}
        self.connection_innovations = {}
    
    def add_node_gene(self, start_node_id, end_node_id): # node is added between start and end
        self.last_node_id += 1
        



class Connection_Gene_History:
    def __init__(self):
        self.innovation_number = 0
        self.history = {}
    
    def get_innovation_number(self, connection):
        if connection not in self.history:
            self.innovation_number += 1
            self.history[connection] = self.innovation_number
        
        return self.history[connection]
    
    def __contains__(self, connection):
        return connection in self.history

class Node_Gene_History:
    def __init__(self):
        self.innovation_number = -1
        self.history = {}
        self.node_levels = {}
    
    def get_innovation_number(self, connection, src_node):
        
        if connection not in self.history:
            self.innovation_number += 1
            self.history[connection] = self.innovation_number
            print(self.innovation_number)
            self.node_levels[self.innovation_number] = self.node_levels[src_node] + 1
        
        return self.history[connection], self.node_levels[self.innovation_number]
    
    def add_initial_node(self, node_level, node_id=None):
        if node_id is not None:
            if self.innovation_number < node_id:
                self.innovation_number = node_id
            self.history[str(self.innovation_number)] = node_id
        else:
            self.innovation_number += 1
            self.history[str(self.innovation_number)] = self.innovation_number
        
        self.node_levels[self.innovation_number] = node_level
        
        if node_id is not None:
            return node_id, node_level
        return self.innovation_number, node_level
    
    def __contains__(self, connection):
        return connection in self.history



class Node_Gene:
    def __init__(self, src_node, dst_node, node_gene_history:Node_Gene_History, add_initial=False, add_initial_node_level=None, initial_node_id=None):
        connection = str(src_node)+'->'+str(dst_node)
        if add_initial:
            self.innovation_number, self.node_level = node_gene_history.add_initial_node(add_initial_node_level, node_id=initial_node_id)
        else:
            self.innovation_number, self.node_level = node_gene_history.get_innovation_number(connection, src_node)
          
        #self.src_node = src_node
        #self.dst_node = dst_node

class Connection_Gene:
    def __init__(self, in_node, out_node, weight, is_disabled, connection_gene_history:Connection_Gene_History):
        connection = str(in_node)+'->'+str(out_node)
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.is_disabled = is_disabled
        self.innovation_number = connection_gene_history.get_innovation_number(connection)
        
            
class Genotype:
    def __init__(self, node_genes, connection_genes,
                 node_gene_history:Node_Gene_History, connection_gene_history:Connection_Gene_History,
                 mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob, mutate_add_node_prob_large_pop, mutate_add_link_prob):
        
        self.node_genes = node_genes
        self.connection_genes = connection_genes
        self.node_gene_history = node_gene_history
        self.connection_gene_history = connection_gene_history
        self.mutate_weight_prob = mutate_weight_prob
        self.mutate_weight_perturb = mutate_weight_perturb
        self.mutate_weight_random = mutate_weight_random
        self.mutate_add_node_prob = mutate_add_node_prob
        self.mutate_add_node_prob_large_pop = mutate_add_node_prob_large_pop
        self.mutate_add_link_prob = mutate_add_link_prob
        
    
    def mutate(self):
        # mutate weight
        for connection_gene in self.connection_genes:
            if np.random.rand() < self.mutate_weight_prob:
                if np.random.rand() < self.mutate_weight_perturb:
                    connection_gene.weight += np.random.normal()
                else:
                    connection_gene.weight = np.random.normal()
        
        # mutate add node
        if np.random.rand() < self.mutate_add_node_prob:
            self.add_node()

        # mutate add link
        if np.random.rand() < self.mutate_add_link_prob:
            self.add_connection()
            
    def add_node(self):
        # select a random connection gene
        while True:
            connection_gene = np.random.choice(self.connection_genes)
            if not connection_gene.is_disabled:
                break 
            
        connection_gene.is_disabled = True
        
        # add node gene
        node_gene = Node_Gene(connection_gene.in_node, connection_gene.out_node, self.node_gene_history)
        self.node_genes.append(node_gene)
        
        # add connection genes, first weight is 1.0, second is the one of the original
        connection_gene1 = Connection_Gene(connection_gene.in_node, node_gene.innovation_number, 1.0, False, self.connection_gene_history)
        connection_gene2 = Connection_Gene(node_gene.innovation_number, connection_gene.out_node, connection_gene.weight, False, self.connection_gene_history)
        self.connection_genes.append(connection_gene1)
        self.connection_genes.append(connection_gene2)
    
    def add_connection(self):
        while True:
            # first select random source node
            src = np.random.choice(self.node_genes)
            
            # select destination node where level is higher than source
            dsts = [node_gene for node_gene in self.node_genes if node_gene.node_level > src.node_level]
            if len(dsts) == 0: # case when no more connection available
                continue
            dst = np.random.choice(dsts)
            
            
            # check if connection already exists
            connection = str(src.innovation_number)+'->'+str(dst.innovation_number)
            if not connection in self.connection_gene_history:
                break
        
        # add connection gene
        connection_gene = Connection_Gene(src.innovation_number, dst.innovation_number, np.random.normal(), False, self.connection_gene_history)
        self.connection_genes.append(connection_gene)
    
    
    def _crossover_genes(self, fitness_self, fitness_other, genes_self, genes_other):
        more_fit = genes_self if fitness_self > fitness_other else genes_other
        less_fit = genes_self if fitness_self < fitness_other else genes_other
        
        # create new node genes
        more_fit_innovations = set([gene.innovation_number for gene in more_fit])
        less_fit_innovations = set([gene.innovation_number for gene in less_fit])
        
        overlap = np.array(list(more_fit_innovations.intersection(less_fit_innovations)))
        disjoint = np.array(list(more_fit_innovations - less_fit_innovations)) # disjoint and excess of more fit
        
        mask = np.random.choice([True, False], len(overlap))
        print(mask)
        if len(disjoint) > 0:
            from_more_fit = np.concatenate([overlap[mask], disjoint])
        else:
            from_more_fit = overlap[mask]
        print('from_more_fit', from_more_fit)
        from_less_fit = overlap[~mask]
        print('from_less_fit', from_less_fit)
        genes1 = [node_gene for node_gene in more_fit if node_gene.innovation_number in from_more_fit]
        genes2 = [node_gene for node_gene in less_fit if node_gene.innovation_number in from_less_fit]
     
        genes1.extend(genes2)
        return genes1
    
    def crossover(self, other, fitness_self, fitness_other):
        node_genes = self._crossover_genes(fitness_self, fitness_other, self.node_genes, other.node_genes)
        print('conn genes')
        connection_genes = self._crossover_genes(fitness_self, fitness_other, self.connection_genes, other.connection_genes)
        # create new connection genes
        print('aa', connection_genes)
        return Genotype(node_genes, connection_genes, self.node_gene_history, self.connection_gene_history, self.mutate_weight_prob, self.mutate_weight_perturb, self.mutate_weight_random, self.mutate_add_node_prob, self.mutate_add_node_prob_large_pop, self.mutate_add_link_prob)
    
    def distance(self, other):
        pass
    
    def __str__(self):
        return str(self.node_genes) + '\n' + str(self.connection_genes)

# %%
# test crossover
node_gene_history = Node_Gene_History()
connection_gene_history = Connection_Gene_History()
node_genes1 = [Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0), 
              Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0),
              Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=10**10)] # output node

node_genes2 = [Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=0), 
              Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=1),
              Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=10**10, initial_node_id=3)] # output node

connection_genes1 = [
    Connection_Gene(0, 2, 0.3, False, connection_gene_history), 
    Connection_Gene(1, 2, 0.4, False, connection_gene_history)
    ]

connection_genes2 = [
    Connection_Gene(0, 2, -0.2, False, connection_gene_history), 
    Connection_Gene(1, 2, 0.7, False, connection_gene_history)
    ]

genotype1 = Genotype(node_genes1, connection_genes1, node_gene_history, connection_gene_history, mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob, mutate_add_node_prob_large_pop, mutate_add_link_prob)
genotype2 = Genotype(node_genes2, connection_genes2, node_gene_history, connection_gene_history, mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob, mutate_add_node_prob_large_pop, mutate_add_link_prob)
genotype3 = genotype1.crossover(genotype2, 1.0, 0.9)

# %%
genotype3.node_genes

# %%
for gene in genotype1.node_genes:
    print(gene.innovation_number)

for gene in genotype1.connection_genes:
    print(gene.weight)

for gene in genotype2.connection_genes:
    print(gene.weight)
    
for gene in genotype3.connection_genes:
    print(gene.weight)
    

# %%
node_gene_history = Node_Gene_History()
connection_gene_history = Connection_Gene_History()

exit()
# %%
node_gene_history.history

# %%
a = Node_Gene(1, 3, node_gene_history)
a.innovation_number

# %%
class NeuralNetwork():
    def __init__(self, genotype:Genotype) -> None:
        pass


