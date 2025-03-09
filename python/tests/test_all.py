import os
os.environ['MPLBACKEND'] = 'Agg'

import pytest
from pydm import pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from test_python_implementation import pdm_test

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

plt.plot(freq,theta)
plt.savefig('theta.png')


# period_range = (1.5, 7.5, 1000)
# start = time.time()
# periods, thetas, best_period = pdm_test(t, y, period_range)
# print(f"computed in {time.time()-start}")

# plt.plot(periods,thetas)
# plt.savefig('theta2.png')

start = time.time()
freq, theta = pdm(t, y, min_freq, max_freq, n_freqs, n_bins)
print(f"computed in {time.time()-start}")


plt.plot(freq,theta)
plt.savefig('theta3.png')