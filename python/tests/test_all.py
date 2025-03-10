import pytest
from pydm import pdm as rust_pdm
from pdmpy import pdm as c_pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


resolution = int(1e7)
t = np.linspace(0,100,resolution)

y = np.linspace(0,10,resolution) + np.sin(t)
# t = pd.date_range(
#     start='2022-03-10',
#     end='2022-03-15',
#     periods=resolution
# ).values

min_freq = 1.5
max_freq = 7.5
n_freqs = int(1e2)
n_bins = 10
float_dates = t.astype('datetime64[ns]').astype('int64').astype('float64')
#float_dates = float_dates-float_dates[0]
start = time.time()
freq, theta = rust_pdm(t,y,min_freq,max_freq,n_freqs, n_bins)
print(f"pydm computed in {time.time()-start}")

plt.figure()
plt.plot(freq,theta)
plt.savefig('theta_rust.png')

freq_step = (max_freq-min_freq)/n_freqs
start = time.time()

freq, theta = c_pdm(t, y, f_min = min_freq, f_max = max_freq, delf = freq_step, nbin = n_bins)
print(f"py-pdm computed in {time.time()-start}")

plt.figure()
plt.plot(freq,theta)
plt.savefig('theta_c.png')