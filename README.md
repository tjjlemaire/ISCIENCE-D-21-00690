# Description

This repository contains the Python scripts allowing to generate the figures of the **morphoSONIC** paper by Lemaire et al. (2021).

## Contents

- `fig**.py`: scripts used to generate the paper figures that result from model simulations.
- `LICENSE`: license file.
- `config.py`: module specifying the path to the data root directory.

# Requirements

- Python 3.6+
- NEURON 7.x (https://neuron.yale.edu/neuron/download/)
- PySONIC package (https://github.com/tjjlemaire/PySONIC)
- MorphoSONIC package (https://github.com/tjjlemaire/MorphoSONIC)

# Installation

- Install a Python distribution
- Install a NEURON distribution
- Download the PySONIC and MorphoSONIC code bases from their repositories, and follow the README instructions to install them as packages.

# Usage

## Create a data directory

First, you must create a directory on your machine to hold the generated data. Once this is done, open the `config.py` and specify the full path to your data directory (replacing `None`).

## Generating the data

To run the necessary underlying simulations and render a specific figure (or sub-figure), just call the associated python script, e.g. for figure xxx:

```
python figxxx.py
```

Upon completion, the figure panels should appear as matplotlib figures. Additionally, you can ue the `-s` option to save them as PDF files in the *figs* sub-folder.

Be aware that the raw simulations as well as intermediate computation results are stored upon script executation, in order to avoid re-running simulations twice. The **total size of entire dataset exceeds 50 GB**, so make sure to plan for enough disk space. Complementarily, **some scripts are optimized to distribute simulations among across CPUs** (using the `--mpi` option), so you might want to run them on a **high-performance, multi-core machine**.

The generated dataset should be split between 5 sub-folders in the indicated output directory:

- `benchmarks`: contains primary results and divergence maps files of two-compartment benchmark simulations 
- `bundle`: contains results files of the nerve bundle simluation
- `fields`: contains results from acoustic fields computations
- `figs`: output folder containing PDFs of the generated figures
- `modulation`: contains primary results and intermediate firing rate maps from fiber neuromodulation simulations
- `multiplexing`: contains primary results and firing rate maps files of MUX-LIFUS analysis
- `responses`: contains simulation results from fiber "typical" simulations
- `SDcurves`: contains intermediate SD curves data from fiber threshold simulations.

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire@epfl.ch).

# License

At the moment, access to this project is granted strictly for the purpose of reviews of the associated paper. As such, the code should not be re-used or distributed - see the LICENSE file for details.