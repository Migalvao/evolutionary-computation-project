import pandas as pd
import numpy as np
from random import randint, random, sample
from operator import itemgetter
# DISCLAIMER: This code was obtained (and adapted) from Evolutionary Computation's practical classes material

# Fitness evaluation
def best_pop(populacao):
    populacao.sort(key=itemgetter(1),reverse=True)
    return populacao[0]

def avg_fitness(pop, fitness_function):
    return sum([fitness_function(indiv[0]) for indiv in pop]) / len(pop)

# Variation operators: ------ > swap mutation
def swap_mutation(cromo, prob_muta):
    if  random() < prob_muta:
        comp = len(cromo) - 1
        copia = cromo[:]
        i = randint(0, comp)
        j = randint(0, comp)
        while i == j:
            i = randint(0, comp)
            j = randint(0, comp)
        copia[i], copia[j] = copia[j], copia[i]
        return copia
    else:
        return cromo

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

# Simple [permutation] Evolutionary Algorithm		
def sea_perm(numb_generations,size_pop, size_cromo, prob_mut,  prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func, gera_pop):

    populacao = gera_pop(size_pop,size_cromo)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    best = best_pop(populacao)
    
    all_best = []
    all_fitness = []
    
    for gen in range(numb_generations):
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
        
        candi = best_pop(populacao)
        if(candi[1]<best[1]):
            best = candi
            
        all_best.append(candi[1])
        all_fitness.append(avg_fitness(populacao, fitness_func))
            
    return best, all_best, all_fitness

def run_multiple(filename,numb_runs,numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func, gera_pop):
    bests = []
    avgs = []

    for i in range(numb_runs):
        best, all_best, all_avg_fitness = sea_perm(numb_generations,size_pop, size_cromo,prob_mut, prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func, gera_pop)
        bests.append(all_best)
        avgs.append(all_avg_fitness)


    bests = np.array(bests)
    avgs = np.array(avgs)

    bests_avg = [sum(bests[:,i])/ len(bests[:,i]) for i in range(numb_generations)]
    avgs_avg = [sum(avgs[:,i])/ len(avgs[:,i]) for i in range(numb_generations)]

    df = pd.DataFrame(np.transpose([bests_avg, avgs_avg]), columns=["best", "average"])
    
    df.to_csv(filename, index=False)