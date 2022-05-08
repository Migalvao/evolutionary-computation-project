import pandas as pd
import numpy as np
from random import randint, random, sample
from operator import itemgetter

# Initialize population
def gera_pop(size_pop,size_cromo):
    return [(gera_indiv(size_cromo),0) for i in range(size_pop)]

def gera_indiv(size_cromo):
    # random initialization
    indiv = [randint(0,1) for i in range(size_cromo)]
    return indiv

# Fitness evaluation
def best_pop(populacao):
    populacao.sort(key=itemgetter(1),reverse=True)
    return populacao[0]

def avg_fitness(pop, fitness_function):
    return sum([fitness_function(indiv[0]) for indiv in pop]) / len(pop)

# Variation operators: Binary mutation	    
def muta_bin(indiv,prob_muta):
    # Mutation by gene
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_bin_gene(cromo[i],prob_muta)
    return cromo

def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g

# Parents Selection: tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def one_tour(population,size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=True)
    return pool[0]

# Survivals Selection: elitism
def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

# Simple [Binary] Evolutionary Algorithm		
def sea(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop,size_cromo)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

    all_best = [best_pop(populacao)[1]]
    all_fitness = [avg_fitness(populacao, fitness_func)]

    for i in range(numb_generations):
        # sparents selection
        mate_pool = sel_parents(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao,descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]    
        
        all_best.append(best_pop(populacao)[1])
        all_fitness.append(avg_fitness(populacao, fitness_func))

    return best_pop(populacao), all_best, all_fitness

def run_multiple(filename,numb_runs,numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    bests = []
    avgs = []

    for i in range(numb_runs):
        best, all_best, all_avg_fitness = sea(numb_generations,size_pop, size_cromo,prob_mut, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
        bests.append(all_best)
        avgs.append(all_avg_fitness)


    bests = np.array(bests)
    avgs = np.array(avgs)

    bests_avg = [sum(bests[:,i])/ len(bests[:,i]) for i in range(numb_generations)]
    avgs_avg = [sum(avgs[:,i])/ len(avgs[:,i]) for i in range(numb_generations)]

    df = pd.DataFrame(np.transpose([bests_avg, avgs_avg]), columns=["best", "average"])
    
    df.to_csv(filename, index=False)