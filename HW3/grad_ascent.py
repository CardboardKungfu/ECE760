import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_ascent(X, y, learning_rate, iterations):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    for _ in range(iterations):
        gradient = 0
    
        for i, row in enumerate(X):
            x_i = row.T
            gradient += (y[i] - sigmoid(np.dot(theta_hat, x_i))) * x_i
        
        theta_hat += learning_rate * gradient
    
    return theta_hat

data = np.loadtxt("titanic_data.csv", delimiter=',', skiprows=1)

y = data[:, 0] # response vector (survived or not)

fare_col_normed = (data[:,-1] / np.max(data[:,-1])).reshape(-1, 1) # normalize the fare column, ideally to avoid exp overflow
X = np.hstack((data[:, 1:-1], fare_col_normed))

# print("X")
# print(X)
learning_rate = 0.001
iterations = 1000

# Call gradient ascent function
coefficients = gradient_ascent(X, y, learning_rate, iterations)

print("Coefficients:", coefficients)

