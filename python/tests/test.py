from phasedm import pdm as rust_pdm
from pdmpy import pdm as c_pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from phasedm import beta_test


resolution = int(1e4)
t = np.linspace(0, 20, resolution)

y = np.sin(t) + np.random.normal(0, 1, resolution)

sigma = np.zeros(resolution)

# t = pd.date_range(
#     start='2022-03-10 12:00:00',
#     end='2022-03-10 12:00:20',
#     periods=resolution
# ).values

plt.plot(t, y)

min_freq = 0.05
max_freq = 1
n_bins = 10
n_freqs = int(1e4)

sig_theta = beta_test(resolution, n_bins, 0.0001)
print(sig_theta)

start = time.time()
freq, theta = rust_pdm(
    t, y, min_freq, max_freq, n_freqs, sigma=sigma, n_bins=n_bins, verbose=1
)
pydm_time = time.time() - start
print(f"pydm computed in {pydm_time}")

# Find the best period
best_freq = freq[np.argmin(theta)]
print(f"True period: {2*np.pi}, Detected period: {1/best_freq}")

# Plot results

plt.figure()
plt.plot(freq, theta)
plt.axvline(1 / (2 * np.pi), color="red", linestyle="--", label="True Frequency")
plt.axvline(best_freq, color="green", linestyle=":", label="Detected Period")
plt.axvline(best_freq / 2, color="red", linestyle=":", label="Harmonic Period")

plt.axhline(sig_theta, color="blue", linestyle="--", label="Significance Threshold")
plt.xlabel("Frequency")
plt.ylabel("PDM Statistic")
plt.title("Phase Dispersion Minimisation Results")
plt.legend()
plt.show()

freq_step = (max_freq - min_freq) / n_freqs
start = time.time()
freq, theta = c_pdm(t, y, sigma, min_freq, max_freq, freq_step, n_bins)
pdmpy_time = time.time() - start
print(f"py-pdm computed in {pdmpy_time}")

# Find the best period
best_freq = freq[np.argmin(theta)]
print(f"True period: {2*np.pi}, Detected period: {1/best_freq}")

# Plot results
plt.figure()
plt.plot(freq, theta)
plt.axvline(1 / (2 * np.pi), color="red", linestyle="--", label="True Frequency")
plt.axvline(best_freq, color="green", linestyle=":", label="Detected Period")
plt.xlabel("Frequency")
plt.ylabel("PDM Statistic")
plt.title("Phase Dispersion Minimisation Results")
plt.legend()
plt.show()

print(f"{pdmpy_time/pydm_time} x speed-up")
