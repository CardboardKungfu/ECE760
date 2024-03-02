import numpy as np
import pandas as pd

data_path = 'titanic_data.csv'
titanic_data = pd.read_csv(data_path)

X = titanic_data.drop('Survived', axis=1).values
y = titanic_data['Survived'].values

# Problem 4.1: Transform features into binary variables
pclass = X[:, 0]
sex = X[:, 1]
age = X[:, 2]
siblings_spouse = X[:, 3]
parents_children = X[:, 4]
fare = X[:, 5]

bin_X = np.where(pclass > 1, 1, 0) # Split pclass between 1 and 2/3
bin_X = np.vstack((bin_X, sex)) # Sex is already binary, so add it
bin_X = np.vstack((bin_X, np.where(age > np.median(age), 1, 0))) # Split based on median age
bin_X = np.vstack((bin_X, np.where(siblings_spouse > 0, 1, 0))) # If any siblings or spouse, then 1, otherwise 0
bin_X = np.vstack((bin_X, np.where(parents_children > 0, 1, 0))) # If any parents or children, then 1, otherwise 0
bin_X = np.vstack((bin_X, np.where(fare > np.median(fare), 1, 0))) # Split based on median fare