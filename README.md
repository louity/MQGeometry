# MQGeometry

Mutli-layer QG model on arbitrary geometries implemented in Pytorch, [paper](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1715/egusphere-2023-1715.pdf) accepted at GMD.

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

## Acknowledgments

Louity acknowledges the support of the ERC EU project 856408-STUOD for the development of this package.
