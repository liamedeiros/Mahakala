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
from jax import lax


EE = 4.8032e-10
KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
HPL = 6.6261e-27
GNEWT = 6.6743e-8


def solve_specific_intensity(emissivity, absorptivity, dt, L_unit):
    """
    Solve the radiative transfer equation for the specific intensity given
    - emissivity: the invariant emissivity at each step
    - absorptivity: the invariant absorptivity at each step
    - dt: the time step at each step
    - L_unit: the length unit, which is multiplied into the step size

    Returns:
    - I_new: the final "image" specific intensity
    - I_list: the specific intensity computed for each step
    """
    nsteps, npx = emissivity.shape

    def solve_one_step(I_nu, i):
        dI = - dt[i-1, :]*L_unit * (emissivity[i, :] - (absorptivity[i]*I_nu))
        I_nu += dI
        return I_nu, dI

    I_nu = jnp.zeros(npx)
    I_nu, dIs = lax.scan(solve_one_step, I_nu, jnp.arange(nsteps - 1, 0, -1))

    return I_nu, dIs


def synchrotron_coefficients(Ne, Theta_e, B, pitch_angle, nu,
                             invariant=True, rescale_nu=1.):
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
        rescaled_nu = nu * rescale_nu
        emissivity = emissivity / rescaled_nu**2.
        absorptivity = absorptivity * rescaled_nu

    emissivity = emissivity.at[jnp.isnan(emissivity)].set(0)
    absorptivity = absorptivity.at[jnp.isnan(absorptivity)].set(0)

    return emissivity, absorptivity
