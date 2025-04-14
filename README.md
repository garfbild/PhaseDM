# PhaseDM: Phase Dispersion Minimisation for Python

PhaseDM is a high-performance implementation of the Phase Dispersion Minimisation algorithm for Python, built with Rust. This package offers significant advantages over existing implementations like pdm-py, making it an ideal choice for time series analysis.

## Features

- **High Performance**: Up to 100x faster than pure Python implementations and 10x than single threaded c implementions through parallelization with Rayon
- **Better Compatibility**: No Visual Studio development tools required
- **Enhanced DateTime Support**: Full support for `datetime[ns]` format (not available in pdm-py)
- **Beta Statistic**: Support for statistical analysis using Beta distribution
<p align="center">
<img src="Timer_comparison.png" width="720" alt="Alt text">
</p>

### Prerequisites
- Python 3.8+

### Option 1: Install from PyPI
```bash
pip install phasedm
```

### Option 2: Install from source

#### Step 1: Install uv (fast Python package installer)
Follow the installation instructions at https://docs.astral.sh/uv/getting-started/installation/

#### Step 2: Create a virtual environment in the repository
```bash
uv venv
```

#### Step 3: Activate the virtual environment
```bash
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

#### Step 4: Install dependencies
```bash
uv pip install maturin numpy matplotlib
```

#### Step 5: Build and install the package
```bash
maturin develop --release
```

## Usage

```python
from phasedm import pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from phasedm import beta_test


resolution = int(1e4)

t = np.linspace(0, 20, resolution)

y = np.sin(t) + np.random.normal(0, 1, resolution)

sigma = np.zeros(resolution)

# t = pd.date_range(
#     start='2022-03-10 12:00:00',
#     end='2022-03-10 12:00:20',
#     periods=resolution
# ).values

plt.plot(t, y)

min_freq = 0.05
max_freq = 1
n_bins = 10
n_freqs = int(1e4)

sig_theta = beta_test(resolution, n_bins, 0.0001)
print(f"Significant theta {sig_theta}")

start = time.time()
freq, theta = rust_pdm(
    t, y, min_freq, max_freq, n_freqs, sigma=sigma, n_bins=n_bins, verbose=1
)
pydm_time = time.time() - start
print(f"pydm computed in {pydm_time}")

# Find the best period
best_freq = freq[np.argmin(theta)]
print(f"True period: {2*np.pi}, Detected period: {1/best_freq}")

# Plot results

plt.figure()
plt.plot(freq, theta)
plt.axvline(1 / (2 * np.pi), color="red", linestyle="--", label="True Frequency")
plt.axvline(best_freq, color="green", linestyle=":", label="Detected Period")
plt.axvline(best_freq / 2, color="red", linestyle=":", label="Harmonic Period")

plt.axhline(sig_theta, color="blue", linestyle="--", label="Significance Threshold")
plt.xlabel("Frequency")
plt.ylabel("PDM Statistic")
plt.title("Phase Dispersion Minimisation Results")
plt.legend()
plt.show()
```
## Comparison with Other Implementations

| Feature | phasedm | pdm-py |
|---------|------|--------|
| Performance | Up to 10x faster | Baseline |
| DateTime Support | ✅ | ❌ |
| Significance Testing | ✅  | ❌ |
| Dependencies | No VS dev tools | Requires Visual Studio tools on Windows |
| PDM2 | Planned | ❌ |

## Technical Details
The main crates we use are
- **Maturin**: Builds and publishes Rust-based Python packages
- **PyO3**: Enables Rust to interact with Python code and objects
- **NumPy**: Efficient numerical operations in Python
- **ndarray**: Rust library for n-dimensional arrays
- **Rayon**: Provides data parallelism for Rust

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References
- Stellingwerf https://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=34
- Schwarzenberg-Czerny https://iopscience.iop.org/article/10.1086/304832
- PY-PDM https://pypi.org/project/Py-PDM/ 

## License
MIT Licence