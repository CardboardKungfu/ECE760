import numpy as np
import pandas as pd
from scipy.stats import norm

# Find entropy, given that it is distributed normally

data_path = 'titanic_data.csv'
titanic_data = pd.read_csv(data_path)

X = titanic_data.drop('Survived', axis=1).values
y = titanic_data['Survived'].values
y = y.reshape(-1, 1) # Make into column vector

# Problem 4.1: Transform features into binary variables
pclass,sex,age,siblings_spouse,parents_children,fare = X[:, 0].reshape(-1,1), X[:, 1].reshape(-1,1), X[:, 2].reshape(-1,1), X[:, 3].reshape(-1,1), X[:, 4].reshape(-1,1), X[:, 5].reshape(-1,1)
bin_X = np.where(pclass > 1, 1, 0).reshape(-1, 1) # Split pclass between 1 and 2/3
bin_X = np.hstack((bin_X, sex)) # Sex is already binary, so add it
bin_X = np.hstack((bin_X, np.where(age > np.median(age), 1, 0))) # Split based on median age
bin_X = np.hstack((bin_X, np.where(siblings_spouse > 0, 1, 0))) # If any siblings or spouse, then 1, otherwise 0
bin_X = np.hstack((bin_X, np.where(parents_children > 0, 1, 0))) # If any parents or children, then 1, otherwise 0
bin_X = np.hstack((bin_X, np.where(fare > np.median(fare), 1, 0))) # Split based on median fare

# 4.2 Find mutual information
def entropy(x_vec):
    H = 0
    unique, counts = np.unique(x_vec, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    for x in unique:
        H += (counts_dict[x] / x_vec.size) * np.log2(1 / (counts_dict[x] / x_vec.size))
    return H

def cond_entropy(x_vec, y_vec):
    H = 0
    x_unique, x_counts = np.unique(x_vec, return_counts=True)
    y_unique, y_counts = np.unique(y_vec, return_counts=True)
    x_dict = dict(zip(x_unique, x_counts))
    y_dict = dict(zip(y_unique, y_counts))

    for x in x_unique:
        for y in y_unique:
            cond_x_and_y = np.sum((x_vec == x) & (y_vec == y)) / np.sum(y_vec == y)
            if cond_x_and_y != 0:  # Avoid division by zero
                H += (x_dict[x] / x_vec.size) * (y_dict[y] / y_vec.size) * np.log2( 1 / (cond_x_and_y))
    return H

def mutual_information(x, y):
    return entropy(x) - cond_entropy(x, y)

# Example from notes to make sure I created my functions correctly
def notes_example():
    X = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    ])

    y = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    # Make it so that columns are features and rows are samples
    X = X.T
    y = y.reshape(-1, 1)

    print("x1:")
    x1 = X[:, 0].reshape(-1, 1)
    print(entropy(x1))
    print(cond_entropy(x1, y))
    print(mutual_information(x1, y))

    print("x2:")
    x2 = X[:, 1].reshape(-1, 1)
    print(entropy(x2))
    print(cond_entropy(x2, y))
    print(mutual_information(x2, y))

    print("x3:")
    x3 = X[:, 2].reshape(-1, 1)
    print(entropy(x3))
    print(cond_entropy(x3, y))
    print(mutual_information(x3, y))

for column in bin_X.T:
    print(mutual_information(column.T, y))

# 4.3