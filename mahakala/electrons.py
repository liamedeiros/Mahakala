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

KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
MP = 1.6726e-24
EC = 4.8032e-10
HPL = 6.6261e-27
GNEWT = 6.6743e-8

def rlow_rhigh_model(dens, u, beta, r_low=1, r_high=40, electron_gamma=4./3, ion_gamma=5./3):
    """
    Compute dimensionless electron temperature given the following parameters:
    - dens: density
    - u: internal energy
    - beta: plasma beta
    - r_low: lower limit of the integral (default = 1)
    - r_high: upper limit of the integral (default = 40)

    Returns:
    - Theta_e: dimensionless electron temperature
    """

    T_ratio = (r_high * beta**2 + r_low) / (1 + beta**2)
    t_electron = CL**2 * (MP * u * (electron_gamma - 1.) * (ion_gamma - 1.))/(dens * ((ion_gamma - 1.) + (electron_gamma - 1.) * T_ratio))

    return t_electron / (ME*CL*CL)
