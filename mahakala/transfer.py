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


KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
EC = 4.8032e-10
HPL = 6.6261e-27


def emission_coefficient(Ne, t_electron, B, nu, beta, angle):

    thetae = (KB * t_electron)/( ME * CL**2)

    nuc = 2.79925e6*B       # Eq [2]
    nus = (2./9.)*nuc*thetae*thetae*np.sin(angle)

    X=nu/nus # Eq [56]

    var = np.exp(-np.power(X,1./3.))

    synemiss = Ne*nus*np.power(jnp.sqrt(X)+np.power(2., 11./12.)*np.power(X,1./6.),2.)/(special.kn(2,1./thetae))
    return synemiss  * var * np.sqrt(2) * np.pi * e_c**2 /(3 * CL)



def absorption_coefficient(t_electron, je, nu, Beta):

    B_nu = (2*HPL*nu**3/c**2)/(pow(np.e,HPL*nu/(KB*t_electron)) - 1)

    return je/B_nu


