
# Mahakala

Copyright (C) 2024 Aniket Sharma, Lia Medeiros, George N. Wong, CK Chan, et al.

Mahakala is a python + jax based implementation of a radiative ray-tracing algorithm for arbitrary spacetimes. See https://arxiv.org/abs/2304.03804 for more detail.

## Installation

Mahakala can be installed using pip

```bash
git clone https://github.com/liamedeiros/Mahakala.git
cd Mahakala
pip install .
```

## Demos

Demonstration notebooks can be found in the ```demos/``` folder within the root of this repository.

```demos/shadows.ipynb``` shows examples of how to find and plot geodesic trajectories as well as how to find the shadow boundary

```demos/grmhd_simple.ipynb``` shows how to use the helper utility functions in Mahakala to create an image from a GRMHD snapshot file (GRMHD snapshot not included in git repo)

```demos/grmhd_detailed.ipynb``` shows how to create an image from a GRMHD snapshot file going step by step tracing the geodesics, sampling the GRMHD file, and solving the radiative transfer equation (GRMHD snapshot not included in git repo)

Note that we do not include the GRMHD snapshot in the git repo due to the large file size, but please see [AthenaK](https://github.com/IAS-Astrophysics/athenak) for the GRMHD code that produced the files used in the demos

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
