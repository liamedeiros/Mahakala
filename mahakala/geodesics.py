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
from jax import jit, jacfwd, vmap, lax
from jax.numpy.linalg import inv


def initialize_geodesics_at_camera(bhspin, inclination, distance, ll, ul,
                                   pixels_per_side, camera_type='grid'):
    '''
    @ Brief Returns list of photon positions and wavevectors across image.

    bhspin - normalized black hole spin parameter
    inclination - in degrees
    distance - between the image and the coordinate origin (in GM/c^2)
    ll - lower limit of the image plane
    ul - upper limit of the image plane
    pixels_per_side - linear resolution of the returned image array
    camera_type - default=grid or equator (e.g., for studying cross sections)
    '''

    s0_x, s0_v = get_initial_grid(inclination, distance, ll, ul,
                                  pixels_per_side, camera_type)

    return initial_condition(s0_x, s0_v, bhspin)


def Nullify(metric, bhspin, p=1):
    '''
    !@brief This functions takes the metric as a parameter, and then nullifies
    the velocity vector to make it a null geodesic.
    '''
    assert p > 0

    @jit
    def nullify(x, v):

        g = metric(x, bhspin)
        A = v[:p] @ g[:p, :p] @ v[:p]
        b = v[p:] @ g[p:, :p] @ v[:p]
        C = v[p:] @ g[p:, p:] @ v[p:]

        d1, d2 = quadratic(A, b, C)
        S = jnp.select([d1 > 0, d2 > 0], [d1, d2], jnp.nan)

        return jnp.concatenate([v[:p], v[p:] / S])

    return nullify


def quadratic(A, b, C):
    '''
    !@brief Used by Nullify

    This is an intermediate function that makes sure that
    our velocity vector follows a  null geodesics
    '''
    bb = b * b
    AC = A * C
    dd = jnp.select([~jnp.isclose(bb, AC)], [bb - AC], 0)
    bs = jnp.heaviside(b, 1)
    D = - (b + bs * jnp.sqrt(dd))
    x1 = D / A
    x2 = C / D
    return jnp.minimum(x1, x2), jnp.maximum(x1, x2)


@jit
def metric(x, bhspin):
    '''
    Calculates the Cartesian Kerr metric in Cartesian Kerr-Schild coordinates.
    @param x 1-D jax array with 4 elements corresponding to {t,x,y,z}
    @returns a 4 X 4 two dimensional array reprenting the metric

    '''
    eta = jnp.diag(jnp.array([-1, 1, 1, 1]))
    a = bhspin
    aa = a * a
    zz = x[3]**2.
    kk = 0.5 * (x[1]*x[1] + x[2]*x[2] + zz - aa)
    rr = jnp.sqrt(kk * kk + aa * zz) + kk
    r = jnp.sqrt(rr)
    f = (2.0 * rr * r)/(rr * rr + aa * zz)
    l = jnp.array([1, (r * x[1] + a * x[2])/(rr + aa), (r * x[2] - a * x[1])/(rr + aa), x[3]/r])
    return eta + f * (l[:, jnp.newaxis] * l[jnp.newaxis, :])


def get_camera_pixel(inclination, distance, radius, angle):

    size = np.size(radius)
    x = jnp.ones(size) * np.cos(angle) * radius  # angle MUST be in radians
    y = jnp.ones(size) * np.sin(angle) * radius
    z = jnp.ones(size) * 0.

    origin_BH = Image_to_BH(0, 0, 0, inclination, distance)
    temp_coord = _perpendicular([x, y])

    init_BH = Image_to_BH(x, y, z, inclination, distance)
    perp_BH = Image_to_BH(temp_coord[0], temp_coord[1], jnp.zeros(1), inclination, distance)

    vec1 = - init_BH.T + origin_BH
    vec2 = perp_BH - init_BH

    k_vec = jnp.cross(vec1, vec2.T)
    s0_x = jnp.array([jnp.zeros(size), init_BH[0].flatten(), init_BH[1].flatten(), init_BH[2].flatten()])
    s0_v = jnp.array([jnp.ones(size), k_vec.T[0].flatten(), k_vec.T[1].flatten(), k_vec.T[2].flatten()])

    return s0_x, s0_v


def get_initial_grid(inclination, distance, fov_lower, fov_upper, spacing,
                     camera_type):
    """
    Compute positions and (not-necessarily normalized) wavevectors for photons
    in the image plane assuming either 'grid' or 'equator' camera type.
    - inclination: inclination of observer in degrees
    - distance: distance of observer from BH in GM/c^2
    - fov_lower: left and bottom edge of field of view in GM/c^2
    - fov_upper: right and top edge of field of view in GM/c^2
    - spacing: linear resolution of the grid
    - camera_type: 'grid' or 'equator'

    If camera_type is 'grid', then the function returns a grid of photons in
    the image plane. When the camera_type is 'equator', the function returns
    a line of photons in the equatorial plane.

    Returns:
    - s0_x: array containing the position 4-vectors of all the photons
    - s0_v: array containing the 4-wavevectors of all the photons
    """

    if camera_type.lower() == 'grid':

        grid_list = np.linspace(fov_lower, fov_upper, 2*spacing + 1)[1::2]

        z = 0 * jnp.ones(len(grid_list)**2)
        x, y = jnp.meshgrid(grid_list, grid_list, indexing='ij')

        x = x.flatten()
        y = y.flatten()

        origin_BH = Image_to_BH(0, 0, 0, inclination, distance)
        temp_coord = _perpendicular([x, y])

        init_BH = Image_to_BH(x, y, z, inclination, distance)
        perp_BH = Image_to_BH(temp_coord[0], temp_coord[1], 0 * jnp.ones(len(grid_list)**2), inclination, distance)

        vec1 = - init_BH.T + origin_BH
        vec2 = perp_BH - init_BH

        k_vec = np.cross(vec1, vec2.T)
        s0_x = np.array([np.zeros(len(grid_list)**2), init_BH[0].flatten(), init_BH[1].flatten(), init_BH[2].flatten()])
        s0_v = np.array([np.ones(len(grid_list)**2), k_vec.T[0].flatten(), k_vec.T[1].flatten(), k_vec.T[2].flatten()])

        return s0_x, s0_v

    elif camera_type.lower() == 'equator':

        grid_list = jnp.linspace(fov_lower, fov_upper, 2*spacing + 1)[1::2]

        # initialize positions
        s0_x = np.zeros((4, len(grid_list)))
        s0_x[1] = distance
        s0_x[2] = grid_list

        # initialize wavevectors
        s0_v = np.ones((4, len(grid_list)))
        s0_v[2] = 0
        s0_v[3] = 0

        return s0_x, s0_v

    else:
        print(f'Unexpected camera type \"{camera_type}\". Please choose either "grid" or "equator"')


def Image_to_BH(x, y, z, i, d):
    i = i * np.pi/180
    x_BH = -y * np.cos(i) + z * np.sin(i) + d * np.sin(i)
    y_BH = x
    z_BH = y * np.sin(i) + z * np.cos(i) + d * np.cos(i)
    return np.array([x_BH, y_BH, z_BH])


def _perpendicular(a):
    b = np.zeros_like(a)
    b[0] = a[0] + a[1]
    b[1] = a[1] - a[0]
    return b


def initial_condition(s0_x, s0_v, bhspin):
    """
    Make initial wavevectors null given positions and orientations of photons.
    - s0_x: photon position in cartesian KS coordinates
    - s0_v: photon wavevectors in cartesian KS coordinates (need not be null)
    """
    make_null = Nullify(metric, bhspin)

    def concatenate_func(x_col, v_col):
        return jnp.concatenate([x_col, make_null(x_col, v_col)])

    return vmap(concatenate_func, in_axes=(1, 1))(s0_x, s0_v)


def geodesic_integrator(N, s0, div, tol, bhspin):
    '''
    JAX implementation of the geodesic integrator.

    ## TODO: make better documentation

    - div: division factor for the timestep
    - tol: tolerance for the timestep

    TODO: change first argument (max_steps) to be optional kwarg
    '''

    def geodesic_step(s0, _):

        # get new timestep
        dt = -(radius_cal(s0[:, :4], bhspin) - radius_EH(bhspin)) / div
        condition = jnp.logical_or(jnp.abs(dt) * div > 1500, jnp.abs(dt) * div < tol)
        condition = jnp.logical_or(condition, jnp.isnan(dt))
        dt = lax.select(condition, jnp.zeros_like(dt), dt)

        # get new state
        new_state = RK4_gen(s0, dt, bhspin)

        # get new timestep
        dt_new = -(radius_cal(new_state[:, :4], bhspin) - radius_EH(bhspin)) / div
        condition = jnp.logical_or(jnp.isnan(dt_new), jnp.abs(dt_new) * div < tol)
        condition = jnp.logical_or(condition, jnp.abs(dt_new) * div > 1500)
        dt_new = lax.select(condition, jnp.zeros_like(dt_new), dt_new)

        # in places where next dt == 0, set new_state = s0 and dt = 0
        condition = dt_new == 0.
        dt = lax.select(condition, jnp.zeros_like(dt), dt)
        condition = jnp.broadcast_to(condition[:, None], s0.shape)
        new_state = lax.select(condition, s0, new_state)

        return new_state, (s0, dt)

    # use scan because next step depends on the last
    _, (states1, final_dt) = lax.scan(geodesic_step, s0, jnp.arange(N))

    # get index where final_dt is zero
    all_zero_mask = jnp.all(final_dt == 0, axis=1)
    first_zero_idx = jnp.argmax(all_zero_mask)  # + 2
    if first_zero_idx < 1:
        first_zero_idx = N
    first_zero_idx += 2

    return states1[:first_zero_idx], final_dt[:first_zero_idx]


def radius_cal(x, bhspin):
    """
    Compute spherical KS radius given a point in cartesian KS coordinates.
    - x: 4-vector position in cartesian KS coordinates
    - bhspin: black hole spin parameter
    """
    R = jnp.sqrt(x[..., 1]**2 + x[..., 2]**2 + x[..., 3]**2)
    return jnp.sqrt((R**2 - bhspin**2 + jnp.sqrt((R**2 - bhspin**2)**2 + 4 * bhspin**2 * x[..., 3]**2)) / 2)


@jit
def RK4_gen(state1, dt, bhspin):

    val = len(state1)
    ans1 = vectorized_rhs(state1, bhspin)

    k1 = jnp.multiply(dt.reshape(val, 1), ans1)
    ans1 = vectorized_rhs(state1 + 0.5 * k1, bhspin)
    k2 = jnp.multiply(dt.reshape(val, 1), ans1)
    ans1 = vectorized_rhs(state1 + 0.5 * k2, bhspin)
    k3 = jnp.multiply(dt.reshape(val, 1), ans1)
    ans1 = vectorized_rhs(state1 + k3, bhspin)
    k4 = jnp.multiply(dt.reshape(val, 1), ans1)

    new_state1 = state1 + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return new_state1


@jit
def vectorized_rhs(s0, bhspin):
    return vmap(rhs, in_axes=(0, None))(s0, bhspin)


@jit
def rhs(state1, bhspin):
    '''
    !@brief Calculates the RHS of the Geodesic equation
    @param state An n * 8 dimensional array where n = number of photons, 8 = 8 = 4 position coordinates + 4 velocity coordinates

    @returns An n * 8 dimensional array where n = number of photons, 8 = 8 = 4 velocity coordinates + 4 acceleration coordinates
    '''

    x = state1[:4]
    v = state1[4:]

    ig = imetric(x, bhspin)
    jg = jit(jacfwd(metric))(x, bhspin)

    a = ig @ (- (jg @ v) @ v + 0.5 * v @ (v @ jg))

    return jnp.concatenate([v,a])


@jit
def imetric(x, bhspin):
    """
    Return the inverse of the metric (i.e., the contravariant metric)
    using linalg.inv, given
    - x: 4-vector position in cartesian KS coordinates
    - bhspin: black hole spin parameter
    """
    return inv(metric(x, bhspin))


def radius_EH(a_spin):
    return 1 + np.sqrt(1-a_spin**2)


def select_photons_integrator(inc, angle, radius, bhspin,
                              distance=1000, max_steps=2000):
    """
    Integrate geodesics for a set of photons with different initial conditions
    - inc: inclination angle of observer in degrees
    - angle: array of azimuthal angles in radians
    - radius: array of radii in GM/c^2
    - bhspin: black hole spin parameter
    - distance: distance of observer from BH in GM/c^2 (default = 1000.)
    - max_steps: maximum number of integration steps (default = 2000)

    Returns the minimum distance of each photon to the coordinate origin
    """
    s0_x, s0_v = get_camera_pixel(inc, distance, radius, angle)
    init_one = initial_condition(s0_x, s0_v, bhspin)
    S, dt = geodesic_integrator(max_steps, init_one, 40, 1e-2, bhspin)
    r = radius_cal(S, bhspin)

    # find last point along geodesic
    maxi = np.argmax(dt, axis=0)
    maxi.at[maxi < 1].set(dt.shape[0] - 1)
    maxi -= 1
    maxi.at[maxi < 0].set(0)

    return r[maxi, np.arange(r.shape[1])]  # returns minimum distance from BH


def find_shadow_bisection(bhspin, inc, num_angles, max_steps=2000,
                          error_allowed=0.001, max_it=40):
    """
    Wrapper for the find_shadow_bisection_angles function that automatically
    generates the angles over which to compute the shadow.
    - bhspin: black hole spin parameter
    - inc: inclination angle of observer in degrees
    - num_angles: number of angles on the image to find the shadow over
    - max_steps: maximum number of integration steps (default = 2000)
    - error_allowed: allowed uncertainty in the shadow radius (default = 0.001)
    - max_it: maximum number of iterations to find the shadow radius (default = 40)
    """
    angles = np.arange(num_angles)/num_angles*2.*np.pi

    radii = find_shadow_bisection_angles(bhspin, inc, angles, max_it=max_it,
                                         error_allowed=error_allowed,
                                         max_steps=max_steps)

    radii = np.append(radii, radii[0])
    angles = np.append(angles, angles[0])

    return angles, radii


def find_shadow_bisection_angles(bhspin, inc, angles, max_steps=2000,
                                 error_allowed=0.001, max_it=40):
    """
    Use bisection method on the "last position" of the geodesic to find the
    boundary of the black hole shadow.
    - bhspin: black hole spin parameter
    - inc: inclination angle of observer in degrees
    - angles: array of image angles in radians
    - max_steps: maximum number of integration steps (default = 2000)
    - error_allowed: allowed uncertainty in the shadow radius (default = 0.001)
    - max_it: max number of iterations to find the shadow radius (default = 40)
    """
    inner = np.zeros_like(angles) + 0.5
    outer = np.zeros_like(angles) + 10
    error = outer - inner

    bisection_limit = 100

    counter = 0
    while np.max(error) > error_allowed and counter < max_it:
        final_mid = select_photons_integrator(inc, angles, (outer-inner)/2+inner, bhspin, max_steps=max_steps)
        fell = np.where(final_mid < bisection_limit)
        got_away = np.where(final_mid >= bisection_limit)
        inner[fell] = (outer[fell]-inner[fell])/2+inner[fell]
        outer[got_away] = (outer[got_away]-inner[got_away])/2+inner[got_away]
        error = outer - inner
        counter += 1

    return inner
