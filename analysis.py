import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import wilcoxon, test_normal_sw, histogram_with_normal
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

def statisical_analysis(filename1, filename2, name1, name2, plot_sw_test=False):
    df1 = pd.read_csv("results\\" + filename1)
    df2 = pd.read_csv("results\\" + filename2)
    
    fitness1 = df1['fitness'].tolist()
    fitness2 = df2['fitness'].tolist()
    generations1 = df1['generation'].tolist()
    generations2 = df2['generation'].tolist()

    data = [test_normal_sw(fitness1), test_normal_sw(fitness2), test_normal_sw(generations1), test_normal_sw(fitness2)]
    labels = ["Fitness " + name1, "Fitness " + name2, "Generations " + name1, "Generations " + name2]

    if plot_sw_test:
        plot_shapiro_wilk(data, 0.05, labels)

    print("Wilcoxon test for fitness: ")
    res = wilcoxon(fitness1, fitness2)
    print(f"Statistical value: {res.statistic}, P-value: {res.pvalue}")
    print("Mean of best fitness for " + name1 + ":", np.mean(fitness1))
    print("Mean of best fitness for " + name2 + ":", np.mean(fitness2))

    print("\nWilcoxon test for generation: ")
    res = wilcoxon(generations1, generations2)
    print(f"Statistical value: {res.statistic}, P-value: {res.pvalue}")
    print("Mean of generation for " + name1 + ":", np.mean(generations1))
    print("Mean of generation for " + name2 + ":", np.mean(generations2))


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

if __name__ == '__main__':
    # compare_averages(["pmx_cross", "order_cross"])
    statisical_analysis("pmx_cross.csv", "order_cross.csv", "PMX", "Order X", plot_sw_test=False)