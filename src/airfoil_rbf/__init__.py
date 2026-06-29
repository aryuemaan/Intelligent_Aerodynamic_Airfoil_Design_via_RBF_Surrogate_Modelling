"""Intelligent Aerodynamic Airfoil Design via RBF surrogates.

Adaptation of Clark (GT2019-91637) to the NREL windAI_bench
OEDI 9k-airfoil CFD dataset (8,996 shapes, 2 AoA, M=0.1, Re=9e6).

Public sub-modules:
    geometry   - CST / Kulfan parameterisation of airfoil surfaces
    features   - duty + aerodynamic-style feature extraction
    data       - streaming dataset builder and quality filters
    models     - RBF surrogate, scaler, feasibility, optimisation
    viz        - plotting helpers
"""

__version__ = "1.0.0"
