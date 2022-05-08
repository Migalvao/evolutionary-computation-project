from sea_bin_sol import *
from jb_sol_3_4 import * 
from utils import *


cross_oper = [one_point_cross, two_points_cross, uniform_cross]
recombination = one_point_cross
numb_runs = 30
numb_generations = 100
size_pop = 100
size_cromo = 100
prob_mut = 0.01
prob_cross = 0.9
tour_size = 3
sel_parents = tour_sel(tour_size)
mutation = muta_bin
elite = 0.02
sel_survivors = sel_survivors_elite(elite)
fitness_func = fitness 

# #Compute results
# data = [None, None, None]
# for i, recombination in enumerate(cross_oper):
#     data[i] = run_best_at_the_end(numb_runs,numb_generations, size_pop,size_cromo,prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
#     data[i].sort()

best_gen, average_pop_gen = sea_for_plot(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
display_stat_1(best_gen,average_pop_gen)   