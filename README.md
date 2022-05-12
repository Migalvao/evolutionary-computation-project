# Evolutionary Computation Project

The goal of this project is to test and compare two different permutation operators for Evolutionary Computation algorithms, Partially Mapped Crossover and Order Crossover.
Miguel Galvão, 2022

### Parameterization

To configure the parameterization, open the file `parameterization.py` and change the values in the variables probs_mut and probs_cross according to the values you wish to compare regarding the probability of mutation and crossover, respectively.
You can also change the variable num_runs to define how many times each combination of values should be tested.
All other variables, such as chromossome size, population size or number of generations can also be changed to adapt the experience setup.
If you wish to run the tests and write the results out to .csv files so you can later compare the results, set the value for the if clause to "True".
If, otherwise, you only wish to compare existing results, set the value for the if clause do "False".

To run, simply run, in the directory of the project,

```sh
python parameterization.py
```

### Run tests

To configure the testing environment, similar to the parameterization part, you should open the `run_tests.py` file and change the necessary variables to run the different tests.
To change the crossover operators being used, simply add (or remove) another call of the function "run_test" with the crossover function, it's name and also the name of the file in which the results should be stored.

To run, simply run, in the directory of the project,

```sh
python run_tests.py
```

### Analyse results

This file applies, for a pair of crossover algorithms, the Wilcoxon signed rank test to assess the statistical similarity of the resulting distributions. In order to also apply and plot the Shapiro-Wilk test for each of the distributions, change the "show_shapiro_wilk" variable in the main function to "True". Similarly, to present a boxplot of the distributions, change the "show_boxplots" variable value to "True" in the main block of code.
To run these tests simply run, in the directory of the project,

```sh
python analysis.py
```
