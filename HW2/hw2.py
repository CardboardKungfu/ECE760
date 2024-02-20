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

# sigma_star = (y - X @ theta) @ np.transpose(y - X @ theta)

# print("sigma star")
# print(sigma_star)


# while(np.linalg.det(sigma_star) < 1):
#       sigma_star = sigma_star + 0.1 * np.identity(sigma_star.size)

theta_hat = np.linalg.inv(X.T @ np.linalg.inv(sigma_star) @ X) @ X.T @ np.linalg.inv(sigma_star) @ y
print(theta_hat)
# print("MLE of Sigma_star:")
# print(sigma_star)
# print("\nMLE of theta:")
# print(theta_hat)
