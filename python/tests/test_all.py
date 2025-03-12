from phasedm import pdm as rust_pdm
from pdmpy import pdm as c_pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


resolution = int(1e4)
t = np.linspace(0, 20, resolution)

y = np.sin(t)
# t = pd.date_range(
#     start='2022-03-10',
#     end='2022-03-11',
#     periods=resolution
# ).values

min_freq = 0.1
max_freq = 1
n_bins = 10

pydm_times = []
pdmpy_times = []
power = []
n_freqs = int(1e4)

repeats = 5
for i in tqdm(np.linspace(1,5,20)):
    power.append(i)
    n_freqs = int(10**(i))
    
    freq_step   = (max_freq-min_freq)/n_freqs
    pydm_run = 0.0
    pdmpy_run = 0.0

    for j in range(repeats):
        start = time.time()
        freq, theta = rust_pdm(t,y,min_freq,max_freq,n_freqs, n_bins, verbose=0)
        pydm_run += time.time()-start

        start = time.time()
        freq, theta = c_pdm(t, y, f_min = min_freq, f_max = max_freq, delf = freq_step, nbin = n_bins)
        pdmpy_run += time.time()-start
        
    pydm_times.append(pydm_run/repeats)
    pdmpy_times.append(pdmpy_run/repeats)


plt.figure()
plt.plot(power,pydm_times)
plt.plot(power,pdmpy_times)
plt.yscale('log')

plt.savefig('timer_comparison.png')

# print(pydm_times)



# freq_step = (max_freq-min_freq)/n_freqs
# start = time.time()

# freq, theta = c_pdm(t, y, f_min = min_freq, f_max = max_freq, delf = freq_step, nbin = n_bins)
# print(f"py-pdm computed in {time.time()-start}")

# plt.figure()
# plt.plot(freq,theta)
# plt.savefig('theta_c.png')