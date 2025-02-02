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

from jax import jit, lax


EE = 4.8032e-10
KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
HPL = 6.6261e-27
GNEWT = 6.6743e-8

#@jit
def solve_specific_intensity(N, synemiss_data, absorption_data, KuUu, dt, M_BH):
    I_new = np.zeros(synemiss_data.shape[1])
    range_values = jnp.array(range(N-1, 0, -1))
    for i in range_values:
        val = (-(dt[i-1, :]) * (GNEWT*M_BH/CL**2) * (synemiss_data[i, :]/abs(KuUu[i, :])**2 - (abs(KuUu)[i, :] * absorption_data[i, :] * I_new)))
        I_new = I_new + val
    return np.array(I_new)


def solve_specific_intensity_new2(N, synemiss_data, absorption_data, KuUu, dt, M_BH):
    range_values = np.arange(N-1, 0, -1)
    val = -(dt[range_values-1, :] * (GNEWT*M_BH/CL**2) * (synemiss_data[range_values, :]/np.abs(KuUu[range_values, :])**2 - (np.abs(KuUu)[range_values, :] * absorption_data[range_values, :] * np.cumsum(val, axis=0))))
    I_new = np.sum(val, axis=0)
    return np.array(I_new)

def solve_specific_intensity_new(N, synemiss_data, absorption_data, KuUu, dt, M_BH):
    I_new = np.zeros(synemiss_data.shape[1])
    for i in range(N-1, 0, -1):
        val = -(dt[i-1, :] * (GNEWT*M_BH/CL**2) * (synemiss_data[i, :]/abs(KuUu[i, :])**2 - (abs(KuUu)[i, :] * absorption_data[i, :] * I_new)))
        I_new += val
    return np.array(I_new)

@jit
def solve_specific_intensity_jax(N, synemiss_data, absorption_data, KuUu, dt, M_BH):
    
    def body_func(carry, i):
        val = (-(dt[i-1, :]) * (GNEWT*M_BH/CL**2) * (synemiss_data[i, :]/abs(KuUu[i, :])**2 - (abs(KuUu)[i, :] * absorption_data[i, :] * carry)))
        return carry + val, None

    I_new, _ = lax.scan(body_func, jnp.zeros(synemiss_data.shape[1]), jnp.arange(N-1, 0, -1))

    return I_new


def synchrotron_coefficients(Ne, Theta_e, B, pitch_angle, nu):
    """
    Compute thermal synchrotron emissivity and absorptivity given
    - Ne: electron density
    - Theta_e: dimensionless electron temperature
    - B: magnetic field
    - nu: local frequency
    - pitch_angle: pitch angle

    Returns:
    - emissivity: thermal synchrotron in cgs
    - absorptivity: thermal synchrotron in cgs
    """

    nu_max = 1.e12
    Theta_e_min = 0.3

    nuc = EE * B / (2. * np.pi * ME * CL)
    nus = (2. / 9.) * nuc * Theta_e**2 * jnp.sin(pitch_angle)
    X = nu / nus

    var = jnp.exp(- X**(1/3))
    term = jnp.sqrt(X) + 2.0**(11./12) * X**(1./6)

    emissivity = Ne * nus * term**2 / (2.*Theta_e**2.)  # approximation for K2
    emissivity = emissivity * var * jnp.sqrt(2) * jnp.pi * EE**2 / (3.0 * CL)

    emissivity = emissivity.at[jnp.isnan(emissivity)].set(0)
    emissivity = emissivity.at[nu > nu_max].set(0)
    emissivity = emissivity.at[Theta_e < Theta_e_min].set(0)

    # since we should assume jax is using float32, we
    # need to expand for small values of the exponent
    bx = HPL * nu / (ME * CL * CL * Theta_e)
    B_denominator = lax.select(bx < 2.e-3, bx / 24. * (24. + bx * (12. + bx * (4. + bx))), jnp.exp(bx) - 1)
    B_nu = (2. * HPL * nu**3. / B_denominator) / CL**2.

    absorptivity = emissivity / B_nu
    absorptivity = absorptivity.at[jnp.isnan(absorptivity)].set(0)

    return emissivity, absorptivity
