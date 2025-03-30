import numpy as np
import pandas as pd
from astropy.time import Time
from phasedm import pdm, beta_test
import matplotlib.colors as mcolors


def parse_asteroid_lightcurve(file_path):
    """
    Parse asteroid lightcurve data from a text file into numpy arrays.

    Parameters:
    file_path (str): Path to the text file containing the asteroid lightcurve data.

    Returns:
    tuple: (time_array, magnitude_array, error_array) containing the Julian dates,
           magnitudes, and magnitude errors.
    """
    # Initialize lists to store the data
    times = []
    magnitudes = []
    errors = []

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Process each line
    read_data = False
    for line in lines:
        line = line.strip()

        # Check if this is a data line
        if read_data and line.startswith("DATA="):
            # Split by the | delimiter
            parts = line.replace("DATA=", "").split("|")

            # Extract values, ensuring there are 3 parts
            if len(parts) == 3:
                time_val = float(parts[0])
                magnitude_val = float(parts[1])
                error_val = float(parts[2])

                times.append(time_val)
                magnitudes.append(magnitude_val)
                errors.append(error_val)

        # Check for data block markers
        if line == "ENDMETADATA":
            read_data = True
        elif line == "ENDDATA":
            read_data = False

    # Convert lists to numpy arrays
    time_array = np.array(times)
    astropy_time = Time(time_array, format="jd")
    iso_times = astropy_time.iso
    datetime_array = np.array(pd.to_datetime(iso_times))
    magnitude_array = np.array(magnitudes)
    error_array = np.array(errors)

    return datetime_array, magnitude_array, error_array


# Example usage
if __name__ == "__main__":
    # Assuming the input file is named 'asteroid_data.txt'
    file_path = "ALCDEF_10_Hygiea_20250329_174535.txt"

    # Parse the data
    t, y, errors = parse_asteroid_lightcurve(file_path)

    # Print some basic statistics
    print(f"Number of data points: {len(t)}")
    print(f"Time range: {min(t)} to {max(t)}")
    print(f"Magnitude range: {min(y)} to {max(y)}")

    date = "2017-01"
    errors = errors[t > np.datetime64(date)]
    y = y[t > np.datetime64(date)]
    t = t[t > np.datetime64(date)]

    # Optional: Create a simple plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            t,
            y,
            yerr=errors,
            fmt="o",
            markersize=3,
            ecolor="gray",
            elinewidth=1,
            capsize=2,
        )
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Asteroid Lightcurve")
        plt.gca().invert_yaxis()  # Astronomical convention: brighter is lower
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("lightcurve_plot.png")
        print("Plot saved as 'lightcurve_plot.png'")
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")

    min_freq = 0.01
    max_freq = 0.5
    n_bins = 10
    n_freqs = int(1e4)
    sig_theta = beta_test(len(t), n_bins, 0.00001)

    freq, theta = pdm(t, y, min_freq, max_freq, n_freqs, n_bins, verbose=0)

    sorted_indices = np.argsort(theta)
    best_freq = freq[sorted_indices[0]]

    print(f"Detected period: {1/best_freq}")

    plt.figure()
    plt.plot(freq, theta)
    plt.axvline(best_freq, color="green", linestyle=":", label="Detected Period")

    plt.axhline(sig_theta, color="blue", linestyle="--", label="Significance Threshold")
    plt.xlabel("Frequency")
    plt.ylabel("PDM Statistic")
    plt.title("Phase Dispersion Minimisation Results")
    plt.legend()
    plt.show()
    plt.savefig("theta_plot.png")

    ns_timestamps = t.astype(np.int64)
    # Convert nanoseconds to seconds (float)
    seconds_array = ns_timestamps / 1e9
    plt.figure()
    plt.scatter(seconds_array % (1 / (1 * best_freq)), y, alpha=0.5, s=1)
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.title("Asteroid Lightcurve")
    plt.gca().invert_yaxis()  # Astronomical convention: brighter is lower
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase_plot.png")
