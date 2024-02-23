import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def likelihood(y_star, y_hat):
    epsilon = 1e-7 # Define epsilon because log(0) is not defined
    y_hat = np.maximum(np.full(y_hat.shape, epsilon), np.minimum(np.full(y_hat.shape, 1 - epsilon), y_hat))
    log_likelihood = (y_star * np.log(y_hat) + (1 - y_star) * np.log(1 - y_hat))
    return np.mean(log_likelihood)

def gradient_ascent(X, y_true, learning_rate, iterations):
    # use linear regression for best guess of theta_hat
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_true
    
    for _ in range(iterations):
        gradient = 0
    
        for i, row in enumerate(X):
            x_i = row.T
            gradient += (y_true[i] - sigmoid(np.dot(theta_hat, x_i))) * x_i
        
        theta_hat += learning_rate * gradient
    
    return theta_hat

data = np.loadtxt("titanic_data.csv", delimiter=',', skiprows=1)

y = data[:, 0] # response vector (survived or not)
X = data[:, 1:-1]

# fare_col_normed = (data[:,-1] / np.max(data[:,-1])).reshape(-1, 1) # normalize the fare column, ideally to avoid exp overflow
# X = np.hstack((data[:, 1:-1], fare_col_normed))

learning_rate = 0.001
iterations = 1000

coefficients = gradient_ascent(X, y, learning_rate, iterations)

print("Coefficients:", coefficients)

