# MQGeometry

Multi-layer QG model on arbitrary geometries implemented in Pytorch, [paper](https://doi.org/10.5194/gmd-17-1749-2024) accepted at Geoscientific Model Development.

DOI: 10.5194/gmd-17-1749-2024

## Implementation details

This solver solves layered Quasi-geostrophic equations, using the following ingredients:
  - Staggered grid for potential vorticity (PV) and streamfunction (SF)
  - Finite volume PV advection using upwinded WENO reconstructions
  - Elliptic solver using fast discrete sine transform diagonalization for PV inversion
  - Non-rectangular geometries handled with masks and Capacitance matrix method.

Please have a look at the [paper](https://doi.org/10.5194/gmd-17-1749-2024) for more details.

## Requirements

Tested on Intel CPUs and CUDA GPUs.

Python 3.7

```
torch >= 2.0
numpy
matplotlib
```

Also works with ROCm 5.7 and AMD MI210 GPU. When installing torch, simply point to the rocm index
```pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.6```

## Examples

Scripts `double_gyre.py`, `vortex_shear.py`, and `vortex_wall.py` run resp. double gyre, Rankine vortex shear instabiltiy, and vortex-wall interaction numerical experiments.

## Citation

If you use this work please cite

```
@Article{gmd-17-1749-2024,
  AUTHOR = {Thiry, L. and Li, L. and Roullet, G. and M\'emin, E.},
  TITLE = {MQGeometry-1.0: a multi-layer quasi-geostrophic solver on non-rectangular geometries},
  JOURNAL = {Geoscientific Model Development},
  VOLUME = {17},
  YEAR = {2024},
  NUMBER = {4},
  PAGES = {1749--1764},
  URL = {https://gmd.copernicus.org/articles/17/1749/2024/},
  DOI = {10.5194/gmd-17-1749-2024}
}
```
## Acknowledgments

Louity acknowledges the support of the ERC EU project 856408-STUOD for the development of this package.
