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

import mahakala as ma
from mahakala.electrons import rlow_rhigh_model
from mahakala.constants import Msun


def make_image(fluid_model, camera_inclination=60, camera_distance=1000,
               mass_scale=1.e26, M_bh=6.2e9*Msun, r_high=40,
               observing_frequency=230.e9,
               fov=20, resolution=160,
               max_nsteps=10000,
               max_chunk_bytes=None):
    """
    Helper function that wraps the various steps needed to compute an image
    from a fluid model. This function optionally splits the total amount of
    work into chunks to avoid using too much memory on the device.
    - fluid_model: a fluid model object
    - camera_inclination: inclination of the camera in degrees (default = 60)
    - camera_distance: distance of the camera in GM/c^2 (default = 1000)
    - mass_scale: mass scale in grams (default = 1.e26)
    - M_bh: black hole mass in grams (default = 6.2e9*Msun)
    - r_high: r_high parameter in electron temperature model (default = 40)
    - observing_frequency: observing frequency in Hz (default = 230.e9)
    - fov: full field of view in GM/c^2 (default = 20)
    - resolution: number of pixels in each dimension (default = 160)
    - max_nsteps: maximum number of steps for a geodesic (default = 10000)
    - max_chunk_bytes: maximum bytes for one "image segment" evaluation

    Returns:
    - a 2D numpy array of the image specific intensities in cgs
    """

    bhspin = fluid_model.bhspin
    fluid_gamma = fluid_model.fluid_gamma

    args = [camera_inclination, camera_distance, -fov/2., fov/2., resolution]
    s0 = ma.initialize_geodesics_at_camera(bhspin, *args)

    # divide the image into chunks. we assume that the number of pixels in a
    # single chunk should be max_chunk_bytes / 4 / 24 / max_nsteps, where 24
    # is the number of quantities per pixel. this number is probably quite a
    # bit lower than what we need, but it's probably a safe way to start.

    num_pixels_per_chunk = s0.shape[0] + 10
    if max_chunk_bytes is not None:
        num_pixels_per_chunk = int(max_chunk_bytes // 4 // 20 // max_nsteps)

    # variables for tracking image and progress
    I_nu_saved = np.zeros((0))
    lower_limit = 0
    upper_limit = num_pixels_per_chunk

    print(num_pixels_per_chunk, s0.shape[0])

    while lower_limit < s0.shape[0]:

        s0_chunk = s0[lower_limit:upper_limit]
        args = [max_nsteps, s0_chunk, 40, 1e-4, bhspin]
        S, final_dt = ma.geodesic_integrator(*args)

        fluid_scalars = fluid_model.get_fluid_scalars_from_geodesics(S)

        # compute derived quantities
        bsq = fluid_scalars['b'] * fluid_scalars['b']
        beta = fluid_scalars['u'] * (fluid_gamma - 1.) / bsq / 0.5
        beta.at[np.isnan(beta)].set(0.)
        sigma = bsq / fluid_scalars['dens']
        sigma.at[np.isnan(sigma)].set(0.)

        # compute dimensionless electron temperature from rlow/rhigh model
        args = [fluid_scalars['dens'], fluid_scalars['u'], beta]
        Theta_e = rlow_rhigh_model(*args, r_high=r_high)

        # rescale GRMHD to cgs units
        units = fluid_model.get_units(M_bh, mass_scale)
        Ne_in_cgs = units['Ne_unit'] * fluid_scalars['dens']
        B_in_gauss = units['B_unit'] * fluid_scalars['b']
        pitch_angle = fluid_scalars['pitch_angle']
        local_nu = - fluid_scalars['kdotu'] * observing_frequency

        # delete unnecessary fluid varibles
        del fluid_scalars['dens']
        del fluid_scalars['u']
        del fluid_scalars['b']
        del fluid_scalars['kdotu']

        em, ab = ma.synchrotron_coefficients(Ne_in_cgs, Theta_e, B_in_gauss,
                                             pitch_angle, local_nu,
                                             invariant=True,
                                             rescale_nu=1./observing_frequency)

        # apply sigma cutoff
        sigma_cutoff = 100.
        em = em.at[sigma > sigma_cutoff].set(0.)
        ab = ab.at[sigma > sigma_cutoff].set(0.)

        I_nu = ma.solve_specific_intensity(em, ab, final_dt, units['L_unit'])

        I_nu_saved = np.append(I_nu_saved, I_nu)

        # free up memory
        del S
        del final_dt
        del fluid_scalars
        del bsq
        del beta
        del sigma
        del Theta_e
        del Ne_in_cgs
        del B_in_gauss
        del pitch_angle
        del local_nu
        del em
        del ab
        del I_nu

        # get the next chunk
        lower_limit += num_pixels_per_chunk
        upper_limit += num_pixels_per_chunk

    return np.array(I_nu_saved).reshape((resolution, resolution))
