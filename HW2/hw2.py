import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize

y = np.array([
    [110], 
    [140], 
    [180], 
    [190]])

X = np.array([
    [180, 150], 
    [150, 175], 
    [170, 165], 
    [185, 210]])

theta = np.linalg.inv(X.T @ X) @ X.T @ y

residuals = y - X @ theta

RSS = np.sum(residuals ** 2) / 2
n = X.shape[0]  # number of observations
k = X.shape[1]  # number of coefficients
degrees_of_freedom = n - k
variance_of_errors = RSS / degrees_of_freedom

sigma_star = variance_of_errors * np.linalg.inv(X.T @ X)

print("Sigma_star:")
print(sigma_star)

theta_hat = np.linalg.inv(X.T @ np.linalg.inv(sigma_star) @ X) @ X.T @ np.linalg.inv(sigma_star) @ y
print(theta_hat)