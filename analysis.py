import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pingouin import wilcoxon as wilcoxon
from statistics import test_normal_sw, box_plot
plt.rcParams["figure.figsize"] = (10,4)

def plot_shapiro_wilk(data, confidence_level, labels):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    x = np.arange(len(data))
    confidence_level_line = [confidence_level] * len(data)

    ax1.scatter(x, [data[i].statistic for i in range(len(data))])
    ax2.scatter(x, [data[i].pvalue for i in range(len(data))])
    ax2.plot(x, confidence_level_line, label=f"Confidence level {confidence_level}")
    ax2.legend(loc="upper left",ncol=1, prop={'size': 8})

    fig.suptitle('Shapiro-Wilk (normal) test', fontsize=14)
    ax1.set_title('Statistical Value', fontsize=10)
    ax2.set_title('P-value', fontsize=10)
    ax1.set_xticks(x)
    ax1.tick_params(axis='x', labelrotation = 0)
    ax1.set_xticklabels(labels, fontdict={'fontsize': 7})
    ax2.set_xticks(x)
    ax2.tick_params(axis='x', labelrotation = 0)
    ax2.set_xticklabels(labels, fontdict={'fontsize': 7})

    plt.show()

def statisical_analysis(filename1, filename2, name1, name2, show_boxplot=False, plot_sw_test=False):
    df1 = pd.read_csv("results\\" + filename1)
    df2 = pd.read_csv("results\\" + filename2)
    
    fitness1 = df1['fitness'].tolist()
    fitness2 = df2['fitness'].tolist()
    generations1 = df1['generation'].tolist()
    generations2 = df2['generation'].tolist()

    if plot_sw_test:
        data = [test_normal_sw(fitness1), test_normal_sw(fitness2), test_normal_sw(generations1), test_normal_sw(fitness2)]
        labels = ["Fitness " + name1, "Fitness " + name2, "Generations " + name1, "Generations " + name2]

        plot_shapiro_wilk(data, 0.05, labels)

    # FITNESS
    res = wilcoxon(fitness1, fitness2, correction=False).iloc[0]
    print("Wilcoxon test for fitness: ")
    print(f"Statistical value: {res['W-val']}, P-value: {res['p-val']}, Effect Size: {res['RBC']}")
    print("Mean of best fitness for " + name1 + ":", np.mean(fitness1), "Median: ", np.median(fitness1))
    print("Mean of best fitness for " + name2 + ":", np.mean(fitness2), "Median: ", np.median(fitness2))
    
    if show_boxplot:
        box_plot([fitness1, fitness2], ["Fitness " + name1, "Fitness " + name2])

    # GENERATIONS
    res = wilcoxon(generations1, generations2, correction=False).iloc[0]
    print("\nWilcoxon test for generation: ")
    print(f"Statistical value: {res['W-val']}, P-value: {res['p-val']}, Effect Size: {res['RBC']}")
    print("Mean of generation for " + name1 + ":", np.mean(generations1), "Median: ", np.median(generations1))
    print("Mean of generation for " + name2 + ":", np.mean(generations2), "Median: ", np.median(generations2))
    
    if show_boxplot:
        box_plot([generations1, generations2], ["Generations " + name1, "Generations " + name2])

if __name__ == '__main__':
    show_boxplots = False
    show_shapiro_wilk = False
    statisical_analysis("pmx_cross.csv", "order_cross.csv", "PMX", "Order X", show_boxplot=show_boxplots, plot_sw_test=show_shapiro_wilk)