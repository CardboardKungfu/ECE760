import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

def log_likelihood(y_star, y_hat):
    l_likelihood = (y_star * np.log(y_hat) + (1 - y_star) * np.log(1 - y_hat))
    return np.sum(l_likelihood)

def gradient_ascent(X, y, display_weights=True):
    weights = np.zeros(X.shape[1])
    likelihoods = []

    # Run Gradient Ascent
    for i in range(max_iterations):
        y_hat = sigmoid(np.dot(X, weights))
        # gradient = np.sum((y - y_hat) * X.T, axis=1)
        gradient = np.dot(X.T, y - y_hat)
        weights += learning_rate * gradient

        # Show weights at different iterations to see how fast or slow convergence is
        if i % 100 == 0 and display_weights:
            print(f"Weights at {i} iterations: ")
            print(weights)

        likelihood = log_likelihood(y, y_hat)
        likelihoods.append(likelihood)
    return weights, likelihoods

def predict(X, weights, threshold = 0.5):
    probabilities = sigmoid(np.dot(X, weights))
    bin_hats = np.array(list(map(lambda x: 1 if x > threshold else 0, probabilities)))
    return bin_hats

def accuracy(X, y, theta):
    predictions = predict(X, theta)
    return (predictions == y).mean()

data_path = 'titanic_data.csv'
titanic_data = pd.read_csv(data_path)

X = titanic_data.drop('Survived', axis=1).values
y = titanic_data['Survived'].values

min_val = np.min(X)
max_val = np.max(X)

X = (X - min_val) / (max_val - min_val)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.15, random_state=0)
x_train = add_intercept(x_train)
x_test = add_intercept(x_test)

learning_rate = 0.001
max_iterations = 3000

weights, likelihoods = gradient_ascent(x_train, y_train, display_weights=False)

# Prediction based on the testing set
y_hat = predict(x_train, weights)

# Print out values
print('Final Weights: ')
print(weights)
print('MLE of theta hat: ')
print(np.max(likelihoods))

# Test based on my own new feature
new_feature = np.array([1, 1, 0, 24, 0, 0, 70]).reshape(-1, 1)
new_y_hat = predict(new_feature.T, weights.reshape(-1, 1))
print(new_y_hat) # 1 = survive, 0 = go down with the ship

# # Find tau
from scipy.stats import norm

def compute_tau(alpha, features, inv_fisher_information):
    # Compute X * inv_fisher_information
    X_inv_fisher = np.dot(features, inv_fisher_information)

    # Compute X^T * (X * inv_fisher_information)
    X_transpose_X_inv_fisher = np.dot(features.T, X_inv_fisher)

    # Compute the value of tau using the inverse CDF of the standard normal distribution
    tau = norm.ppf(alpha/2, loc=0, scale=np.sqrt(X_transpose_X_inv_fisher))

    return tau


def compute_fisher_information(X, weights):
    sigmoid_vals = sigmoid(np.dot(X, weights))
    W = np.diag(sigmoid_vals * (1 - sigmoid_vals))
    fisher_information = np.dot(X.T, np.dot(W, X))
    return fisher_information

# Compute Fisher Information matrix
fisher_information = compute_fisher_information(X, weights)
inv_fish_inf = np.linalg.inv(fisher_information) # Compute inverse Fisher Information matrix

# Now, compute tau using the corrected Fisher Information matrix
tau = compute_tau(0.05, X, inv_fish_inf)
print("tau")
print(tau)
