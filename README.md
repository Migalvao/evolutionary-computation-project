# Evolutionary Computation Project

The goal of this project is to test and compare two different permutation operators for Evolutionary Computation algorithms.

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

### Compare operators

To configure the testing environment, similar to the parameterization part, you should open the `main.py` file and change the necessary variables to run the different tests.
Just like in the previous section, change the value of the if clause according to wether or not you wish to run the tests or simply compare the results.
To change the crossover operators being used, simply add (or remove) another call of the function "run_test" with the crossover function, it's name and also the name of the file in which the results should be stored.

To run, simply run, in the directory of the project,

```sh
python main.py
```
