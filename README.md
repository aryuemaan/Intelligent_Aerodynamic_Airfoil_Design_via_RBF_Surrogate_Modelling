# Intelligent Aerodynamic Airfoil Design — RBF Surrogate Framework
<img width="2752" height="1304" alt="image" src="https://github.com/user-attachments/assets/381f9446-5a99-498b-b158-e598ab354a43" />

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready implementation of the **Clark (GT2019-91637) Radial Basis Function (RBF) inverse aerodynamic design framework**, adapted for the **NREL WindAI Bench 9k Airfoils CFD Dataset**.

The framework builds fast surrogate models capable of predicting aerodynamic performance and performing inverse airfoil design using Radial Basis Function interpolation. It provides an efficient alternative to computationally expensive CFD simulations while supporting optimization, forward prediction, and inverse aerodynamic design workflows.

---

# Overview

The framework learns relationships between:

* Airfoil geometry
* Surface pressure distribution
* Angle of attack
* Aerodynamic coefficients

It supports two primary workflows:

**Forward Prediction**

Geometry + Flow Conditions → Aerodynamic Performance

**Inverse Design**

Desired Aerodynamic Performance → Airfoil Geometry

The complete pipeline is designed for large-scale CFD datasets while maintaining constant memory usage through streaming data processing.

---

# Performance

Training Dataset

* 8,996 airfoils
* 2 angles of attack
* 17,992 CFD samples

| Metric                  | RMSE    | Relative Error |
| ----------------------- | ------- | -------------- |
| Lift Coefficient (CL)   | 0.00304 | 0.15%          |
| Drag Coefficient (CD)   | 0.00017 | 0.82%          |
| Moment Coefficient (CM) | 0.00043 | 1.28%          |

The forward surrogate consistently achieves sub-1% relative prediction error for all aerodynamic coefficients.

---

# Features

## Forward Aerodynamic Prediction

Predicts:

* Lift coefficient (CL)
* Drag coefficient (CD)
* Moment coefficient (CM)

using

* CST airfoil geometry
* Angle of attack

---

## Inverse Airfoil Design

Design an airfoil directly from:

* Desired angle of attack
* Pressure distribution style
* Target aerodynamic characteristics

The framework predicts both geometry and expected aerodynamic performance.

---

## Aerodynamic Optimization

Supports optimization objectives including:

* Maximum Lift-to-Drag Ratio (L/D)
* Target Lift Coefficient
* Custom objective functions

Optimization is performed using Differential Evolution over CST parameter space.

---

## Large Dataset Processing

Designed for the official NREL WindAI Bench dataset.

Features include:

* Streaming HDF5 loader
* Constant memory usage
* Approximately 2 GB RAM during preprocessing
* Compatible with datasets exceeding 50 GB

---

## Synthetic Demonstration Mode

A complete demonstration pipeline using synthetic airfoils allows users to evaluate the framework without downloading the full CFD dataset.

---

# Quick Start

## Requirements

* Python 3.10+
* Git
* Bash (Git Bash on Windows or Linux/macOS Terminal)
* Minimum 8 GB RAM

---

## Clone Repository

```bash
git clone https://github.com/aryuemaan/airfoil-rbf-surrogate.git
cd airfoil-rbf-surrogate
```

---

## Create Environment

```bash
bash setup_env.sh
```

---

## Run Synthetic Demonstration

```bash
bash run_pipeline.sh --demo
```

This command

* generates synthetic airfoils
* trains the surrogate model
* evaluates prediction accuracy
* produces parity plots in the `figures/` directory

---

# Inverse Design Example

```bash
export PYTHONPATH=src

source .venv/Scripts/activate
# Linux/macOS:
# source .venv/bin/activate

python -m airfoil_rbf design \
    --aoa 4 \
    --style -2.5 0.3 -0.8 0.3
```

---

# Optimization Example

Maximize Lift-to-Drag Ratio

```bash
python -m airfoil_rbf optimize \
    --aoa 4 \
    --objective ld
```

---

# Using the Official NREL Dataset

## Download Dataset

Install AWS CLI.

```text
https://aws.amazon.com/cli/
```

Download the dataset:

```bash
bash download_data.sh
```

The script downloads

```
s3://nrel-pds-windai/aerodynamic_shapes/2D/9k_airfoils/
```

No AWS credentials are required.

---

## Inspect Dataset

```bash
python -m airfoil_rbf inspect
```

If necessary, adjust dataset key names in

```
config.yaml
```

---

## Run Full Training Pipeline

```bash
bash run_pipeline.sh
```

Typical runtime:

30–60 minutes on modern desktop hardware.

---

# Framework Architecture

```text
Official NREL WindAI Dataset (52.7 GB)
                │
                ▼
        io_oedi.py
Streaming HDF5 Data Loader
                │
                ▼
      build_dataset.py
Extract Cp, CST, Style Features
                │
                ▼
      assemble_xy.py
Create Training Matrices
                │
                ▼
          filter.py
Quality Control and Filtering
                │
                ▼
          train.py
Cross Validation and RBF Training
                │
                ▼
        evaluate.py
Performance Evaluation
                │
                ▼
 design.py            optimize.py
```

---

# Technical Details

## Surface Pressure Coefficient

The NREL dataset stores conservative flow variables.

Pressure coefficient is computed using

```text
V/a∞ = |momentum| / density

V/V∞ = (V/a∞) / M∞

Cp = 1 - (V/V∞)^2
```

This approach avoids uncertain energy normalization while maintaining high accuracy for the dataset's freestream Mach number.

---

## CST Geometry Parameterization

Default configuration

* Order = 10
* 22 CST coefficients
* 11 upper surface
* 11 lower surface

Features

* Direct loading from dataset
* No CST fitting error
* Configurable polynomial order

---

## Radial Basis Function Model

The surrogate uses

* Gaussian kernel
* Automatic epsilon tuning
* Cross-validation
* Configurable center selection

Default settings

* Maximum centers: 6000
* Scalable interpolation
* Efficient prediction

---

# Repository Structure

```text
airfoil_rbf_project/

├── README.md
├── LICENSE
├── config.yaml
├── requirements.txt
├── setup_env.sh
├── download_data.sh
├── run_pipeline.sh
│
├── src/
│   └── airfoil_rbf/
│       ├── cli.py
│       ├── config.py
│       ├── io_oedi.py
│       ├── synthetic.py
│       ├── utils.py
│       ├── training.py
│       │
│       ├── geometry/
│       ├── features/
│       ├── data/
│       ├── models/
│       └── viz/
│
├── tests/
├── scripts/
│
├── data/
│   └── airfoil_cfd_9k/
│       └── synthetic_9k.h5
│
└── figures/
    └── perf_parity.png
```

---

# Configuration

Example configuration

```yaml
cst:
  n_order: 10

flow:
  mach_inf: 0.1
  cp_method: incompressible

rbf:
  direction: forward
  max_centres: 6000

filter:
  rmse_threshold: 0.005

  cl_range:
    - -1.0
    - 3.0

  cd_range:
    - 0.0
    - 0.5
```

---

# Testing

Run the complete test suite

```bash
PYTHONPATH=src pytest -q
```

Run selected modules

```bash
PYTHONPATH=src pytest \
tests/test_cst.py \
tests/test_rbf.py \
tests/test_style.py \
-q
```

---

# Recent Improvements

* Support for the official NREL WindAI Bench HDF5 dataset layout
* Direct loading of CST coefficients from the dataset
* Streaming preprocessing for datasets larger than 50 GB
* Improved Windows compatibility for HDF5 file handling
* Synthetic demonstration mode for rapid testing
* Forward prediction accuracy below 1% relative error

---

# Citation

If you use this repository in your research, please cite the following references.

```bibtex
@inproceedings{clark2019rbf,
  title={A Radial Basis Function Surrogate for Inverse Aerodynamic Design},
  author={Clark, C. and others},
  booktitle={ASME Turbo Expo},
  number={GT2019-91637},
  year={2019}
}
```

```bibtex
@dataset{nrel2024windai,
  title={NREL WindAI Bench: 9k Airfoils Aerodynamic Dataset},
  author={NREL},
  year={2024},
  publisher={OEDI Submission 5889}
}
```

---

# License

This project is distributed under the MIT License.

See the `LICENSE` file for details.

---

# Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests.
4. Submit a pull request.

---

# Known Limitations

* The current dataset contains only two angles of attack (4° and 12°), limiting aerodynamic duty-space coverage.
* RBF interpolation scales cubically with the number of centers, making very large models computationally expensive.
* Inverse design accuracy remains lower than forward prediction and continues to be an active area of development.
* On Windows, HDF5 datasets require

```bash
HDF5_USE_FILE_LOCKING=FALSE
```

when accessing large datasets.

---

# Acknowledgements

This implementation is based on the aerodynamic surrogate methodology presented in the Clark GT2019-91637 inverse design framework and is adapted for the NREL WindAI Bench CFD dataset. The project extends the original methodology with streaming data processing, CST-based parameterization, scalable training workflows, and optimization utilities suitable for modern aerodynamic design research.
