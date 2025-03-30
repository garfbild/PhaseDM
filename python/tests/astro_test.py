import numpy as np
import pandas as pd
import requests
import io
import lightkurve as lk
from astroquery.mast import Observations


def download_kepler_data(kic_id=11446443):
    """
    Downloads light curve data for a Kepler target.

    Parameters
    ----------
    kic_id : int or str
        Kepler Input Catalog identifier

    Returns
    -------
    tuple
        (time, flux) numpy arrays
    """
    print(f"Downloading Kepler data for KIC {kic_id}...")
    search_result = lk.search_lightcurve(f"KIC {kic_id}", mission="Kepler")

    if len(search_result) == 0:
        raise ValueError(f"No data found for KIC {kic_id}")

    lc = search_result[0].download()

    # Clean the data - remove NaN values
    mask = ~np.isnan(lc.flux.value)
    time = lc.time.value[mask]
    flux = lc.flux.value[mask]

    print(f"Downloaded {len(time)} data points")

    return time, flux


def download_aavso_data(star_name="R Sct"):
    """
    Downloads light curve data from AAVSO for a variable star.

    Parameters
    ----------
    star_name : str
        Name of the variable star (e.g., 'R Sct', 'Algol')

    Returns
    -------
    tuple
        (time, magnitude) numpy arrays
    """
    print(f"Downloading AAVSO data for {star_name}...")
    star_name_formatted = star_name.replace(" ", "+")

    # Use VSX catalog to get AUID
    try:
        vsx_url = f"https://www.aavso.org/vsx/index.php?view=api.object&ident={star_name_formatted}&data=json"
        response = requests.get(vsx_url)
        data = response.json()

        if "error" in data:
            raise ValueError(f"Star not found: {star_name}")

        auid = data.get("auid", "")

        # Download observations
        url = f"https://www.aavso.org/vsx/index.php?view=api.observations&auid={auid}&data=csv"
        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"Failed to download data: HTTP {response.status_code}")

        df = pd.read_csv(io.StringIO(response.text))

        if df.empty:
            raise ValueError(f"No observations found for {star_name}")

        # Convert to JD and magnitude
        time = df["JD"].values
        mag = df["Magnitude"].values

        # Filter out any invalid data
        mask = ~np.isnan(mag)
        time = time[mask]
        mag = mag[mask]

        print(f"Downloaded {len(time)} observations")
        return time, mag

    except Exception as e:
        print(f"Error downloading AAVSO data: {e}")
        # Provide some sample data as fallback
        print("Generating simple example data")
        np.random.seed(42)

        # Create some sample data
        n_points = 200
        time = np.linspace(0, 500, n_points) + 2455000.0
        period = 140.0
        phase = 2 * np.pi * (time - time[0]) / period
        mag = 9.0 + 1.5 * np.sin(phase) + 0.1 * np.random.randn(n_points)

        return time, mag


# Example usage
if __name__ == "__main__":
    # Download and print a few points from each source
    time_kepler, flux_kepler = download_kepler_data()
    print(f"Kepler data sample:\nTime: {time_kepler[:5]}\nFlux: {flux_kepler[:5]}\n")

    time_aavso, mag_aavso = download_aavso_data()
    print(f"AAVSO data sample:\nTime: {time_aavso[:5]}\nMagnitude: {mag_aavso[:5]}\n")
