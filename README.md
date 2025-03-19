# PhaseDM: Phase Dispersion Minimisation for Python

PhaseDM is a high-performance implementation of the Phase Dispersion Minimisation algorithm for Python, built with Rust. This package offers significant advantages over existing implementations like pdm-py, making it an ideal choice for astronomical time series analysis.

## Features

- **High Performance**: Up to 10x faster than pure Python implementations through Rust parallelization with Rayon
- **Better Compatibility**: No Visual Studio development tools required
- **Enhanced DateTime Support**: Full support for `datetime[ns]` format (not available in pdm-py)
- **Coverage Analysis**: Support for statistical coverage analysis (Coming Soon)
- **Future Development**: F-scoring and PDM2 implementations planned

## Technical Details

PyDM leverages several technologies to achieve its performance and functionality:
- **Maturin**: Builds and publishes Rust-based Python packages
- **PyO3**: Enables Rust to interact with Python code and objects
- **NumPy**: Efficient numerical operations in Python
- **ndarray**: Rust library for n-dimensional arrays
- **Rayon**: Provides data parallelism for Rust

## Installation

### Prerequisites
- Python 3.7+
- Rust compiler (automatically installed by many package managers)

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
from phasedm import pdm as rust_pdm
from pdmpy import pdm as c_pdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

resolution = int(1e4)
t = np.linspace(0, 20, resolution)

y = np.sin(t)
# t = pd.date_range(
#     start='2022-03-10 12:00:00',
#     end='2022-03-10 12:00:20',
#     periods=resolution
# ).values

min_freq = 0.1
max_freq = 1
n_bins = 10
n_freqs = int(1e4)

start = time.time()
freq, theta = rust_pdm(t,y,min_freq,max_freq, n_freqs, n_bins, verbose=1)
pydm_time = time.time()-start
print(f"pydm computed in {pydm_time}")

# Find the best period
best_freq = freq[np.argmin(theta)]
print(f"True period: {2*np.pi}, Detected period: {1/best_freq}")

# Plot results
plt.figure()
plt.plot(freq,theta)
plt.axvline(1/(2*np.pi), color='red', linestyle='--', label='True Frequency')
plt.axvline(best_freq, color='green', linestyle=':', label='Detected Period')
plt.xlabel('Frequency')
plt.ylabel('PDM Statistic')
plt.title('Phase Dispersion Minimisation Results')
plt.legend()
plt.show()
plt.savefig('theta_rust.png')

freq_step = (max_freq-min_freq)/n_freqs
start = time.time()
freq, theta = c_pdm(t, y, f_min = min_freq, f_max = max_freq, delf = freq_step, nbin = n_bins)
pdmpy_time = time.time()-start
print(f"py-pdm computed in {pdmpy_time}")

# Find the best period
best_freq = freq[np.argmin(theta)]
print(f"True period: {2*np.pi}, Detected period: {1/best_freq}")

# Plot results
plt.figure()
plt.plot(freq,theta)
plt.axvline(1/(2*np.pi), color='red', linestyle='--', label='True Frequency')
plt.axvline(best_freq, color='green', linestyle=':', label='Detected Period')
plt.xlabel('Frequency')
plt.ylabel('PDM Statistic')
plt.title('Phase Dispersion Minimisation Results')
plt.legend()
plt.show()
plt.savefig('theta_c.png')

print(f"{pdmpy_time/pydm_time} x speed-up" )
```
## Comparison with Other Implementations

| Feature | PyDM | pdm-py |
|---------|------|--------|
| Performance | Up to 10x faster | Baseline |
| DateTime Support | ✅ | ❌ |
| Coverage Analysis | Coming Soon | ❌ |
| Dependencies | No VS dev tools | Requires Visual Studio tools on Windows |
| F-scoring | Planned | ❌ |
| PDM2 | Planned | ❌ |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Reference
Takes obvious inspiration from PY-PDM https://pypi.org/project/Py-PDM/ and Stellingwerfs orginal version https://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=34
## License
