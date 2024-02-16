import numpy as np
import matplotlib.pyplot as plt

# Generate N i.i.d. Uniform(0, 1) random variables
N = 10
x = np.random.uniform(0, 1, N)

# Plot the histogram
# plt.hist(x, bins=50, density=True, alpha=0.7, color='blue')
# plt.title('Histogram of Uniform(0, 1) Random Variables')
# plt.xlabel('Values')
# plt.ylabel('Probability Density')
# plt.show()

# Function to generate zk's
def generate_zk(N, n, p):
    zk = np.random.binomial(n, p, N)
    return zk

# Plot histograms for different values of p
for p_value in [1/4, 1/2, 3/4]:
    zk = generate_zk(N, 10, p_value)
    plt.hist(zk, bins=np.arange(0, 11), density=True, alpha=0.7, label=f'p = {p_value}')
    plt.title('Histograms of zk for Different p Values')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


# Plot histograms for different values of p
# for p_value in [1/4, 1/2, 3/4]:
#     yi = (x <= p_value).astype(int)
#     zk = np.random.binomial(N, p_value)
#     # yi = generate_yi(N, p_value)
#     # plt.hist(yi, bins=[-0.5, 0.5, 1.5], density=True, alpha=0.7, label=f'p = {p_value}')
#     plt.hist(zk, bins=np.linspace(-0.5, 1.5, 4), density=True, alpha=0.7, label=f'p = {p_value}')
#     plt.title('Histograms of zk for Different p Values')
#     plt.xlabel('Values')
#     plt.ylabel('Probability Density')
#     plt.legend()
#     plt.show()
