from crossover_2019 import pmx_cross, order_cross
# from recombination_operators import cxPartialyMatched
from utils import run_multiple, muta_bin, tour_sel, sel_survivors_elite
from nqueens import fitness, initialize_population
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (10,4)


cross_operators = [pmx_cross, order_cross]
num_runs = 30
num_generations = 400
size_pop = 100
size_cromo = 200
prob_mut = 0.001
prob_cross = 0.8
sel_parents = tour_sel(3)
mutation = muta_bin
sel_survivors = sel_survivors_elite(0.02)
fitness_func = fitness 

# change to run tests
if True:
    filename = f"results\pmx.csv"
    print(f"Running test for PMX")
    run_multiple(filename,num_runs,num_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,pmx_cross,mutation,sel_survivors, fitness_func, initialize_population)
    
    filename = f"results\orderx.csv"
    print(f"Running test for Order crossover")
    run_multiple(filename,num_runs,num_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,order_cross,mutation,sel_survivors, fitness_func, initialize_population)

    print("All done!")

# compare_results(probs_mut, probss)s_cro