import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_likelihood(y_star, y_hat):
    # Avoid undefined log(0) behavior
    # Courtesy of Implementation of Gradient Ascent using Logistic Regression article on Medium
    y_hat = np.maximum(np.full(y_hat.shape, epsilon), np.minimum(np.full(y_hat.shape, 1 - epsilon), y_hat))
    
    # Remember that the MLE is y * log(yh) + (1-y) * log(1-yh)
    likelihood = (y_star * np.log(y_hat) + (1 - y_star) * np.log(1 - y_hat))

    return np.mean(likelihood)

def gradient_ascent(X, y, display_weights=True):
    weights = np.zeros(X.shape[1])
    
    # Run Gradient Ascent
    for i in range(max_iterations):
        y_hat = sigmoid(np.dot(X, weights))
        gradient = np.mean((y - y_hat) * X.T, axis=1)
        weights += learning_rate * gradient

        # Show weights at different iterations to see how fast or slow convergence is
        if i % 100 == 0 and display_weights:
            print(f"Weights at {i} iterations: ")
            print(weights)

        likelihood = log_likelihood(y, y_hat)
        likelihoods.append(likelihood)
    return weights

def predict(X, weights, threshold = 0.5):
    probabilities = sigmoid(np.dot(X, weights))
    bin_hats = np.array(list(map(lambda x: 1 if x > threshold else 0, probabilities)))
    return bin_hats

data = np.loadtxt("titanic_data.csv", delimiter=',', skiprows=1)

y = data[:, 0]
X = data[:, 1:-1]

min_val = np.min(X)
max_val = np.max(X)

X_scaled = (X - min_val) / (max_val - min_val)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.15, random_state=0) 

learning_rate = 0.01
max_iterations = 3000
likelihoods = []

epsilon = 1e-7 # Since log(0) is not defined, create a minimum value epsilon to replace with if necessary

weights = gradient_ascent(x_train, y_train, display_weights=True)

y_hat = predict(x_test, weights)

print('Final Weights: ')
print(weights)
print('MLE of theta hat: ')
print(np.max(likelihoods))