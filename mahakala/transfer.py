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


def specific_intensity_aaa(N,synemiss_data,absorption_data,nu,KuUu,dt):
    I_new = np.zeros(len(synemiss_data[0]))
    I_list = np.zeros((N,2048))
    for i in range(N-1,0,-1):
        val =  (-(dt[i-1,:]) * (G*M_BH/c**2) * (synemiss_data[i,:]/abs(KuUu[i,:])**2 -  (abs(KuUu)[i,:] * absorption_data[i,:] * I_new)))
        I_new = I_new + val

        I_list[N-i,:] = val
    return I_new, I_list

def specific_intensity_bbb(N,synemiss_data,absorption_data,nu,KuUu,dt):
    '''
    !@brief This function calculates the specific intensity of the plasma
    @param N The number of photons
    @param synemiss_data The synchrotron emissivity of the plasma
    @param absorption_data The absorption coefficient of the plasma
    @param nu The frequency of the photon
    @param KuUu The 4-velocity of the photon
    @param dt The time step
    @returns The specific intensity of the plasma
    '''
    I_new = np.zeros(len(synemiss_data[0]))
    I_list = np.zeros((N,2048))
    for i in range(N-1,0,-1):
        val =  (-(dt[i-1,:]) * (G*M_BH/c**2) * (synemiss_data[i,:]/abs(KuUu[i,:])**2 -  (abs(KuUu)[i,:] * absorption_data[i,:] * I_new)))
        I_new = I_new + val

        I_list[N-i,:] = val
    return I_new, I_list

"""
KuUu = local_nu / observing_frequency

EE = 4.8032e-10
KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
HPL = 6.6261e-27
GNEWT = 6.6743e-8

N = len(emissivity)
I_new = np.zeros(len(emissivity[0]))
I_list = np.zeros((N, len(emissivity[0])))
for i in range(N-1,0,-1):
    val =  (-(final_dt[i-1,:]) * (GNEWT*M_bh/CL**2) * (emissivity[i,:]/abs(KuUu[i,:])**2 -  (abs(KuUu)[i,:] * absorptivity[i,:] * I_new)))
    I_new = I_new + val

    I_list[N-i,:] = val
    """

def solve_specific_intensity_newest(emissivity, absorptivity, dt, nu, observing_frequency, L_unit):
    N = len(emissivity)
    I_new = np.zeros(len(emissivity[0]))
    I_list = []
    KuUu = nu / observing_frequency
    for i in range(N-1, 0, -1):
        val = (-(dt[i-1, :]) * L_unit * (emissivity[i, :]/abs(KuUu[i])**2 - (abs(KuUu)[i] * absorptivity[i] * I_new)))
        I_new = I_new + val
        I_list.append(val)
    return I_new, np.array(I_list)


def solve_specific_intensity_mine(invariant_emissivity, invariant_absorptivity, dt):
    ## this is the one that we think is working for now. seems to agree (in terms of code) with others
    ## ... but running tests with ipole now.
    N = invariant_emissivity.shape[0]
    I_invariant = np.zeros(invariant_emissivity.shape[1])
    I_invariant_list = []
    for i in range(N-1, 0, -1):
        I_invariant = I_invariant + (- dt[i-1] * (invariant_emissivity[i] - invariant_absorptivity[i] * I_invariant))
        I_invariant_list.append(I_invariant)
    return np.array(I_invariant_list)


def solve_specific_intensity(invariant_emissivity, invariant_absorptivity, dt, observing_frequency):
    ## this is the one that we think is working for now. seems to agree (in terms of code) with others
    ## ... but running tests with ipole now.
    N = invariant_emissivity.shape[0]
    I_new = np.zeros(invariant_emissivity.shape[1])
    final_I = []
    for i in range(N-1, 0, -1):
        I_new = I_new + (- dt[i-1] * (invariant_emissivity[i,:] - (invariant_absorptivity[i,:] * I_new)))
        final_I.append(I_new)
    return np.array(final_I) * observing_frequency**3.

def solve_specific_intensity_old(N, synemiss_data, absorption_data, KuUu, dt, M_BH):
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


def synchrotron_coefficients(Ne, Theta_e, B, pitch_angle, nu, invariant=True, rescale_nu=1.):
    """
    Compute thermal synchrotron emissivity and absorptivity given
    - Ne: electron density
    - Theta_e: dimensionless electron temperature
    - B: magnetic field
    - nu: local frequency
    - pitch_angle: pitch angle
    - invariant: whether to return invariant coefficients (default = True)
    - rescale_nu: rescale frequency by this factor (default = 1.)

    The rescale_nu factor is useful for dealing with numerical precision
    issues when computing invariant quantities and the frequency is very
    large. Using this variable will change how to interpret the specific
    intensities (i.e., they will *not* need to be rescaled by nu^3). The
    best-guess for a trial rescaling factor is 1/observing_frequency.

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

    emissivity = emissivity.at[nu > nu_max].set(0)
    emissivity = emissivity.at[Theta_e < Theta_e_min].set(0)

    # since we should assume jax is using float32, we
    # need to expand for small values of the exponent
    bx = HPL * nu / (ME * CL * CL * Theta_e)
    B_denominator = lax.select(bx < 2.e-3, bx / 24. * (24. + bx * (12. + bx * (4. + bx))), jnp.exp(bx) - 1)
    B_nu = (2. * HPL * nu**3. / B_denominator) / CL**2.

    absorptivity = emissivity / B_nu

    if invariant:
        #kdotu = nu * rescale_nu
        #emissivity = emissivity  # / nu**2.  # em / nu^2
        #absorptivity = nu**3. * absorptivity  # nu * abs
        emissivity = emissivity / nu**2.
        absorptivity = absorptivity * nu

    emissivity = emissivity.at[jnp.isnan(emissivity)].set(0)
    absorptivity = absorptivity.at[jnp.isnan(absorptivity)].set(0)

    return emissivity, absorptivity
