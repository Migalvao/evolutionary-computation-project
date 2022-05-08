# N Queens Problem
from random import shuffle

def fitness(indiv):
    # This function was adapted from a function available in this public repository:
    #  https://github.com/paulojunqueira/N-Queen-Problem-Evolutionary-Algorithm/blob/master/GA_Nqueens.py
    fit = len(indiv)

    for ia,a in enumerate(indiv, start = 1):
        for ib, b in enumerate(indiv[ia:(len(indiv))], start = ia+1):

            if abs(a-b) == abs(ia-ib):
                fit -= 1

    return fit


# DISCLAIMER: This code was obtained from Evolutionary Computation's practical classes material
# Initialize population
def initialize_population(size_pop,size_cromo):
    return [(gera_indiv_perm(size_cromo),0) for i in range(size_pop)]

def gera_indiv_perm(size_cromo):
    data = list(range(size_cromo))
    shuffle(data)
    return data