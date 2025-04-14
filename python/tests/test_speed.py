from phasedm import pdm as rust_pdm
from pdmpy import pdm as c_pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


resolution = int(1e4)

t = np.linspace(0, 20, resolution)

y = np.sin(t) + np.random.normal(0, 1, resolution)
# t = pd.date_range(
#     start='2022-03-10',
#     end='2022-03-11',
#     periods=resolution
# ).values

min_freq = 0.1
max_freq = 1
n_bins = 10

phasedm_times = []
pdmpy_times = []
power = []
n_freqs = int(1e4)

repeats = 10
for i in tqdm(np.linspace(1, 5, 20)):
    power.append(i)
    n_freqs = int(10 ** (i))

    freq_step = (max_freq - min_freq) / n_freqs
    phasedm_run = 0.0
    pdmpy_run = 0.0

    for j in range(repeats):
        start = time.time()
        freq, theta = rust_pdm(
            t, y, min_freq, max_freq, n_freqs, n_bins=n_bins, verbose=0
        )
        phasedm_run += time.time() - start

        start = time.time()
        freq, theta = c_pdm(
            t, y, f_min=min_freq, f_max=max_freq, delf=freq_step, nbin=n_bins
        )

        pdmpy_run += time.time() - start

    phasedm_times.append(phasedm_run / repeats)
    pdmpy_times.append(pdmpy_run / repeats)


plt.figure(figsize=(10, 6))
plt.plot(power, phasedm_times, marker="o", linewidth=2, label="phasedm (Rust)")
plt.plot(power, pdmpy_times, marker="s", linewidth=2, label="pdmpy (C)")
plt.yscale("log")

# Add title and axis labels
plt.title("Performance Comparison: Rust vs C PDM Implementations", fontsize=14)
plt.xlabel("Number of Frequencies (10^x)", fontsize=12)
plt.ylabel("Execution Time (seconds, log scale)", fontsize=12)

# Add legend
plt.legend(fontsize=10)

# Add grid for better readability
plt.grid(True, which="both", ls="--", alpha=0.3)

# Improve layout
plt.tight_layout()

plt.savefig("Timer_comparison.png", dpi=300)
plt.show()  # Add this if you want to display the plot in addition to saving it
