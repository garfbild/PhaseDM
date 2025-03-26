from phasedm import beta_test
import numpy as np
import matplotlib.pyplot as plt

n_bins = 10
n_freqs = int(1e3)
t = np.linspace(0,1,1000)
p = [beta_test(n_freqs,n_bins,i) for i in t]
plt.plot(t,p)
plt.savefig("beta3.png")