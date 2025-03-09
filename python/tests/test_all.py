import pytest
from pydm import pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


resolution = int(1e3)
t = np.linspace(0,100,resolution)

y = np.linspace(0,10,resolution) + np.sin(t)
t = pd.date_range(
    start='2022-03-10',
    end='2022-03-15',
    periods=resolution
).values

min_freq = 1.5
max_freq = 7.5
n_freqs = int(1e3)
n_bins = 10
float_dates = t.astype('datetime64[ns]').astype('int64').astype('float64')
#float_dates = float_dates-float_dates[0]
start = time.time()
freq, theta = pdm(t,y,min_freq,max_freq,n_freqs, n_bins)
print(f"computed in {time.time()-start}")

plt.figure()
plt.plot(freq,theta)
plt.savefig('theta_rust.png')

start = time.time()
freq, theta = pdm(t, y, min_freq, max_freq, n_freqs, n_bins)
print(f"computed in {time.time()-start}")

plt.figure()
plt.plot(freq,theta)
plt.savefig('theta_c.png')