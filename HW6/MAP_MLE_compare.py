import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import beta, binom


# Parameters
tol = 0.01 # tolerance
alpha = 10     # Parameter of prior distribution
beta_param = 2  # Parameter of prior distribution
# NN = [99, 80, 70, 50, 25, 15, 10, 8, 5, 3]  # Sample sizes we will try
NN = np.array(list(range(10000))) + 1

p_star = 0.01  # True probability of success

UL = p_star + tol
LL = p_star - tol

p_map_n = []
p_map_vals = []
p_mle_n = []
p_mle_vals = []

# Plot distributions of the MAP and the MLE
for N in NN:
    # MAP distribution
    p_values = np.linspace(0, 1, 1000)  # all possible values of p (continuous)
    alpha_prime = N * p_star + alpha  # Parameter of expected posterior distribution
    beta_prime = N * (1 - p_star) + beta_param  # Parameter of expected posterior distribution
    posterior = beta.pdf(p_values, alpha_prime, beta_prime)  # Calculate posterior distribution
    
    # # MLE distribution
    # p_values_mle = np.linspace(0, 1, N + 1)  # all possible values of p (discrete)
    # likelihood = beta.pdf(p_values_mle, 1 + N * p_star, 1 + N * (1 - p_star))  # Calculate likelihood using beta distribution

    # # Normalize likelihood to match the scale of the posterior
    # likelihood = likelihood / np.max(likelihood) * np.max(posterior)

    # # Plot
    # plt.figure()
    # plt.plot(p_values, posterior, 'k', linewidth=4, label='posterior $P(p | \mathbf{X})$')
    # plt.plot(p_values_mle, likelihood, 'b-o', linewidth=2, label='likelihood $P(\mathbf{X} | p)$')
    # plt.xlabel('$p$', fontsize=20)
    # plt.ylabel('', fontsize=20)
    # plt.xticks([0, p_star, 1], ['0', '$p^*$', '1'], fontsize=20)
    # plt.yticks([])
    # plt.title('$N = {}$'.format(N), fontsize=25)
    # plt.legend(fontsize=20, loc='upper left')
    # plt.tight_layout()
    
    # # Save figure
    # plt.savefig(f'images_01/MAPvsMLE_{N}.jpg')

    p_map = p_values[np.argmax(posterior)]
    if (p_map >= LL) & (p_map <= UL):
        p_map_n.append(N)
        p_map_vals.append(p_map)

    # MLE Distribution
    k = np.array(list(range(N))) + 1
    prop = k/N
    log_likelihood = binom.pmf(k, N, p_star)
    p_mle = prop[np.argmax(log_likelihood)]
    if (p_mle >= LL) & (p_mle <= UL):
        p_mle_n.append(N)
        p_mle_vals.append(p_mle)

n_mle = np.min(p_mle_n)
p_mle = p_mle_vals[np.argmin(p_mle_n)]
print(f"n_MLE = {n_mle}, p_MLE = {p_mle:.4f}")
n_map = np.min(p_map_n)
p_map = p_map_vals[np.argmin(p_map_n)]
print(f"n_MAP = {n_map}, p_MAP = {p_map:.4f}")

# plt.show()
