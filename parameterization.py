from crossover_2019 import uniform_cross
from utils import run_multiple, swap_mutation, tour_sel, sel_survivors_elite
from nqueens import fitness, initialize_population
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (10,4)

def compare_results(probs_mut, probs_cross):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    for prob_mut in probs_mut:
        for prob_cross in probs_cross:
            # read from file and plot data
            try:
                df = pd.read_csv(f"results\parameterization\{prob_mut}mut-{prob_cross}cross.csv")
            except FileNotFoundError:
                print(f"File for mutation probability of {prob_mut} and Crossover prob of {prob_cross} not found.")
                continue

            x = df.index

            ax1.legend(loc="lower right",ncol=1)
            ax1.plot(x, df['best'], label=f"{prob_mut}mut-{prob_cross}cross")
            ax2.plot(x, df['average'])

    ax1.set_title('Best')
    ax2.set_title('Average')
    ax1.set_xlabel('Fitness')
    ax2.set_xlabel('Fitness')
    ax1.set_xlabel('Generation')
    ax2.set_xlabel('Generation')

    plt.show()

if __name__ == '__main__':
    numb_generations = 200
    size_pop = 50
    size_cromo = 64
    probs_mut = [0.001, 0.01, 0.05]
    probs_cross = [0.7, 0.8]
    sel_parents = tour_sel(3)
    recombination = uniform_cross
    mutation = swap_mutation
    sel_survivors = sel_survivors_elite(0.02)
    fitness_func = fitness   
    
    num_runs = 5

    # change to run tests
    if True:
        for prob_mut in probs_mut:
            for prob_cross in probs_cross:
                filename = f"results\parameterization\{prob_mut}mut-{prob_cross}cross.csv"
                print(f"Running test for {prob_mut} mutation prob and {prob_cross} crossover prob")
                run_multiple(filename,num_runs,numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func, initialize_population)
    
        print("All done!")

    compare_results(probs_mut, probs_cross)