__copyright__ = """Copyright (C) 2024 Aniket Sharma et al."""
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np


KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
MP = 1.6726e-24
EC = 4.8032e-10
HPL = 6.6261e-27
GNEWT = 6.6743e-8


class GRMHDFluidModel:

    def __init__(self):
        pass

    def get_units(self, M_BH, mass_scale):
        L_unit = GNEWT * M_BH / CL**2
        T_unit = L_unit / CL
        dens_unit = mass_scale / L_unit**3
        Ne_unit = dens_unit / (MP + ME)
        B_unit = CL * np.sqrt(4. * np.pi * dens_unit)

        units = dict(
            L_unit=L_unit,
            T_unit=T_unit,
            dens_unit=dens_unit,
            Ne_unit=Ne_unit,
            B_unit=B_unit
        )

        return units
