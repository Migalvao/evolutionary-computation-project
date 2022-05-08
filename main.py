from crossover_2019 import pmx_cross, order_cross
# from recombination_operators import cxPartialyMatched
from utils import run_multiple, swap_mutation, tour_sel, sel_survivors_elite
from nqueens import fitness, initialize_population
from recombination_operators import cxPartialyMatched
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (10,4)

def compare_results(cross_operators):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    for operator in cross_operators:
        # read from file and plot data
        try:
            df = pd.read_csv(f"results\{operator}.csv")
        except FileNotFoundError:
            print(f"File for {operator} not found.")
            continue

        x = df.index

        ax1.plot(x, df['best'], label=f"{operator}")
        ax2.plot(x, df['average'])
        ax1.legend(loc="lower right",ncol=1, prop={'size': 6})

    ax1.set_title('Best')
    ax2.set_title('Average')
    ax1.set_xlabel('Fitness')
    ax2.set_xlabel('Fitness')
    ax1.set_xlabel('Generation')
    ax2.set_xlabel('Generation')

    plt.show()

cross_operators = [pmx_cross, order_cross]
num_generations = 300
size_pop = 150
size_cromo = 64
prob_mut = 0.05
prob_cross = 0.8
sel_parents = tour_sel(3)
mutation = swap_mutation
sel_survivors = sel_survivors_elite(0.02)
fitness_func = fitness 

num_runs = 30

# change to run tests
if True:
    filename = f"results\pmx_cross.csv"
    print(f"Running test for PMX")
    run_multiple(filename,num_runs,num_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,pmx_cross,mutation,sel_survivors, fitness_func, initialize_population)
    
    filename = f"results\order_cross.csv"
    print(f"Running test for Order crossover")
    run_multiple(filename,num_runs,num_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,order_cross,mutation,sel_survivors, fitness_func, initialize_population)

    print("All done!")

compare_results(["pmx_cross, order_cross"])