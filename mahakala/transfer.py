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
from jax import numpy as jnp
from scipy import special

from jax import jit


KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
EC = 4.8032e-10
HPL = 6.6261e-27
GNEWT = 6.6743e-8

@jit
def solve_specific_intensity(N, synemiss_data, absorption_data, nu, KuUu, dt, M_BH):
    I_new = np.zeros(len(synemiss_data[0]))
    #I_list = np.zeros((N, 2048))
    for i in range(N-1, 0, -1):
        val = (-(dt[i-1, :]) * (GNEWT*M_BH/CL**2) * (synemiss_data[i, :]/abs(KuUu[i, :])**2 - (abs(KuUu)[i, :] * absorption_data[i, :] * I_new)))
        I_new = I_new + val
    return np.array(I_new)


def emission_coefficient(Ne, t_electron, B, nu, beta, angle):

    thetae = (KB * t_electron)/( ME * CL**2)

    # Eq [2]
    nuc = 2.79925e6*B
    nus = (2./9.)*nuc*thetae*thetae*np.sin(angle)

    # Eq [56]
    X=nu/nus

    var = np.exp(-np.power(X,1./3.))
    synemiss = Ne*nus*np.power(jnp.sqrt(X)+np.power(2., 11./12.)*np.power(X,1./6.),2.)/(special.kn(2,1./thetae))

    return synemiss  * var * np.sqrt(2) * np.pi * EC**2 /(3 * CL)


def absorption_coefficient(t_electron, je, nu, Beta):

    B_nu = (2*HPL*nu**3/CL**2)/(pow(np.e,HPL*nu/(KB*t_electron)) - 1)

    return je/B_nu