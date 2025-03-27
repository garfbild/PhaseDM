from phasedm import beta_test, pdm
import numpy as np
import matplotlib.pyplot as plt

resolution = int(1e6)

t = np.linspace(0, 20, resolution)
y = np.random.normal(0, 1, resolution)

min_freq = 0.01
max_freq = 10
n_bins = 10
n_freqs = int(1e4)

sig_theta_2 = beta_test(resolution, n_bins, 0.5)
sig_theta_10 = beta_test(resolution, n_bins, 0.1)
sig_theta_100 = beta_test(resolution, n_bins, 0.01)

freqs, theta = pdm(t, y, min_freq, max_freq, n_freqs, n_bins)

print("0.5:", np.sum(theta < sig_theta_2) / n_freqs)
print("0.1:", np.sum(theta < sig_theta_10) / n_freqs)
print("0.01:", np.sum(theta < sig_theta_100) / n_freqs)
