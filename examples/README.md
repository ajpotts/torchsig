# TorchSig Examples
This folder contains sample Jupyter Notebooks and Markdown files that demonstrate the capabilities of TorchSig.

Install the Python dependencies for most examples from the repository root:

```shell
pip install -e ".[examples]"
```

The GNU Radio WAV example additionally requires GNU Radio 3.10.11 or newer,
which supports NumPy 2. Create its Conda environment and then install TorchSig
into that environment:

```shell
conda env create --file examples/environment.yaml
conda activate torchsig-examples
pip install -e ".[examples]"
```

The same environment file works with the standalone
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
executable when Conda is unavailable:

```shell
micromamba env create --file examples/environment.yaml
micromamba activate torchsig-examples
pip install -e ".[examples]"
```

### Installing GNU Radio with a system package manager

GNU Radio is also available from common system package managers:

```shell
# Debian, Ubuntu, Linux Mint, and Raspberry Pi OS
sudo apt-get update
sudo apt-get install gnuradio python3-venv

# Fedora
sudo dnf install gnuradio

# Arch Linux
sudo pacman -S gnuradio

# Gentoo
sudo emerge net-wireless/gnuradio

# macOS with Homebrew
brew install gnuradio

# macOS with MacPorts
sudo port install gnuradio
```

Distribution repositories do not all provide a sufficiently recent release.
Check the installed version before continuing:

```shell
gnuradio-config-info --version
```

It must report GNU Radio 3.10.11 or newer. If it does not, use the Conda or
micromamba environment above, a container, or a
[source build](https://wiki.gnuradio.org/index.php/InstallingGRFromSource).

System packages install GNU Radio for the package manager's Python. Create the
TorchSig environment with that same interpreter and expose its system packages:

```shell
# Use the Python associated with the GNU Radio package. On Debian/Ubuntu this
# is normally /usr/bin/python3.
/usr/bin/python3 -m venv --system-site-packages .venv-torchsig-examples
source .venv-torchsig-examples/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[examples]"
```

On other platforms, replace `/usr/bin/python3` with the Python installed or
used by that platform's package manager. Verify the final environment rather
than assuming that its compiled extensions are compatible:

```shell
python - <<'PY'
import numpy as np
from gnuradio import gr

print("GNU Radio:", gr.version())
print("NumPy:", np.__version__)
assert tuple(map(int, gr.version().split(".")[:3])) >= (3, 10, 11)
assert int(np.__version__.split(".")[0]) >= 2
PY
```

Do not manually add a system `dist-packages` directory to Python's import
path. GNU Radio contains compiled extensions and must be installed for the
Python and NumPy versions in the active environment. See GNU Radio's
[official installation guide](https://wiki.gnuradio.org/index.php?title=InstallingGR)
for additional platforms and current package availability.

| File | Description  |
| -------- | -----------  |
| getting_started.md | TorchSig overview, description, and terms. |
| bring_your_own_data_npy_example.ipynb | How to read custom NumPy NPZ files into a TorchSig dataset. |
| bring_your_own_data_wav_example.ipynb | How to generate GNU Radio WAV files and read them into a TorchSig dataset. |
| bring_your_own_data_sigmf_example.ipynb | How to read custom SigMF files into a TorchSig dataset. |
| create_dataset_example.ipynb | Creating and customizing datasets. |
| classifier_example.ipynb | Training a PyTorch model on IQ Samples for modulation recognition. |
| defaults_example.ipynb | Demonstrates creating a default dataset and dataloader without any parameterization. |
| detector_example.ipynb| Training a YOLO model on spectrograms for energy detection using spectrograms. |
| filehandler_example.ipynb | How to create and use a custom file handler for writing data to disk in a custom format. |
| reproducibility_example.ipynb | How to create a reproducible dataset and dataloader using random number generator seeding. |
| yaml_dataset_example.ipynb | Saving and loading datasets using YAML configuration files. |
| scripts/ | Provides some useful auxiliary Python scripts. |
| structured_signals/ | Brief overview of structured signals in TorchSig, and illustrative script. |
| transforms/ | Showcases some advanced transforms and how they work. |
