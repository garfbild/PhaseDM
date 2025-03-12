import pytest
from pydm import pdm as rust_pdm
from pdmpy import pdm as c_pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


resolution = int(1e4)
t = np.linspace(0, 20, resolution)

y = np.sin(t)
t = pd.date_range(
    start='2022-03-10 12:00:00',
    end='2022-03-10 12:00:20',
    periods=resolution
).values

min_freq = 0.1
max_freq = 1
n_bins = 10
n_freqs = int(1e4)

start = time.time()
freq, theta = rust_pdm(t,y,min_freq,max_freq, n_freqs, n_bins, verbose=1)
pydm_time = time.time()-start
print(f"pydm computed in {pydm_time}")

plt.figure()
plt.plot(freq,theta)
plt.savefig('theta_rust.png')

min_freq = min_freq*1e9
max_freq = max_freq*1e9
freq_step = (max_freq-min_freq)/n_freqs
start = time.time()
freq, theta = c_pdm(t, y, f_min = min_freq, f_max = max_freq, delf = freq_step, nbin = n_bins)
pdmpy_time = time.time()-start
print(f"py-pdm computed in {pdmpy_time}")

plt.figure()
plt.plot(freq,theta)
plt.savefig('theta_c.png')

print(f"{pdmpy_time/pydm_time} x speed-up" )