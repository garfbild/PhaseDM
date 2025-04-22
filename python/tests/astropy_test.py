import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.timeseries import TimeSeries
from astropy.units import Quantity
from phasedm import testfn, pdm

# Get an example TimeSeries dataset
example_file = get_pkg_data_filename("timeseries/kplr010666592-2009131110544_slc.fits")

# Load the data directly as an Astropy TimeSeries
ts = TimeSeries.read(example_file, format="kepler.fits")

# Extract time, flux and error
times = ts.time  # Extract the numeric values from the Time object
flux = ts["sap_flux"]  # Kepler Simple Aperture Photometry flux
error = ts["sap_flux_err"]  # Flux uncertainty

print(type(flux))
print(flux)

times.format = "jd"

t = testfn(ts.time, flux)
print(t)
# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(np.array(times.datetime64, np.float64), flux, alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Flux (e-/s)")
plt.title("Kepler Lightcurve")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kplr")


min_freq = 4 / int(1e6)
max_freq = 6 / int(1e6)
n_bins = 100
n_freqs = int(1e5)

# print(type(times.datetime64))
# print(type(times.datetime64[0]))
# print(np.isnan(times.datetime64).any())
# print(type(flux.value.astype(np.float64)))
# print(type(flux.value.astype(np.float64)[0]))
# print(np.isnan(flux.value.astype(np.float64)).any())

valid_mask = ~np.isnan(flux.value.astype(np.float64)).astype(bool)

times = times[valid_mask]
flux = flux[valid_mask]

print(np.isnan(flux.value.astype(np.float64)).any())

print(len(times))
print(
    "times",
    (np.array(times.datetime64, np.float64) - np.array(times.datetime64, np.float64)[0])
    / 1e9,
)

print(len(flux))
print("flux", flux)

signal = np.ones(len(times))
freq, theta = pdm(
    times,
    flux,
    min_freq,
    max_freq,
    n_freqs,
    n_bins=n_bins,
    verbose=1,
)

best_freq = freq[np.argmin(theta)]
print(f"Detected period: {1/best_freq}")
print(f"Detected period (days): {(1/best_freq)/60/60/24}")


print(theta)

plt.figure()
plt.plot(freq, theta)
plt.savefig("fig")

plt.figure()
plt.scatter(
    (
        (
            np.array(times.datetime64, np.float64)
            - np.array(times.datetime64, np.float64)[0]
        )
        / 1e9
    )
    % (1 / best_freq),
    flux.value.astype(np.float64),
)
plt.savefig("phase")
