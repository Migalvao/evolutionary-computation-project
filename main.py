from crossover_2019 import pmx_cross, order_cross
from utils import run_multiple, swap_mutation, tour_sel, sel_survivors_elite
from nqueens import fitness, initialize_population
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (10,4)

def compare_averages(cross_operators):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    for operator in cross_operators:
        # read from file and plot data
        try:
            df = pd.read_csv(f"results\{operator}_averages.csv")
        except FileNotFoundError:
            print(f"File for {operator} not found.")
            continue

        x = df.index

        ax1.plot(x, df['best'], label=f"{operator}")
        ax2.plot(x, df['average'])
        ax1.legend(loc="lower right",ncol=1, prop={'size': 10})

    ax1.set_title('Best')
    ax2.set_title('Average')
    ax1.set_xlabel('Fitness')
    ax2.set_xlabel('Fitness')
    ax1.set_xlabel('Generation')
    ax2.set_xlabel('Generation')

    plt.show()

def run_test(operator, operator_name, filename, calculate_average=False):
    filename = f"results\\" + filename
    print(f"Running test for {operator_name}")
    run_multiple(filename,num_runs,num_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,operator,mutation,sel_survivors, fitness_func, initialize_population, calculate_average=calculate_average)

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
    run_test(pmx_cross, "PMX", "pmx_cross", calculate_average=False)
 
    run_test(order_cross, "Order crossover", "order_cross", calculate_average=False)

    print("All done!")

compare_averages(["pmx_cross", "order_cross"])