# ML2020_PR1
First project for the CS-433 Machine Learning course at EPFL, given in 2020.

Repository organisation:

- Necessary code for the mandatory implementations can be found in the implementations.py file. This file contains all necessary code and does not require any other import save numpy.

- The project1 notebook contains our reasoning for the project. In the first part we implement the different mandatory functions and run them on the dataset using cross validation. In the second part we try to improve the predictions of our baseline model, ridge regression, using different techniques (data preprocessing, adding offset, polynomial extension, hyperparameter search).

- The data should be placed in a folder name 'data' at the root of the repository. It should contain files test.csv and train.csv.

- Other files contain code necessary for our development (cross validation, costs, ...).

- The run.py file can be run to obtain our prediction for the challenge, used for submission ID 92768.
