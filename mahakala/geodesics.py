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
from tqdm import tqdm

from jax import numpy as jnp
from jax import jit, jacfwd, vmap, lax
from jax.numpy import dot
from jax.numpy.linalg import inv
##### This will tell whether we are using CPU or GPU #########
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import jax
#jax.config.update("jax_disable_jit", True)

from jax.debug import print as jaxprint


def initialize_geodesics_at_camera(bhspin, inclination, distance, ll, ul, pixels_per_side, camera_type='grid'):
    '''
    @ Brief Return a list of photon positions and wavevectors over the image plane.

    bhspin - normalized black hole spin parameter
    inclination - in degrees
    distance - between the image and the coordinate origin (in GM/c^2)
    ll - lower limit of the image plane
    ul - upper limit of the image plane
    pixels_per_side - linear resolution of the returned image array
    camera_type - default=grid or equator (e.g., for studying cross sections)
    '''

    s0_x, s0_v = get_initial_grid(inclination, distance, ll, ul, pixels_per_side, camera_type)

    return initial_condition(s0_x, s0_v, bhspin)


def Nullify(metric, bhspin, p=1):
    '''
    !@brief This functions takes the metric as a parameter, and then nullifies the velocity vector to make it a null
     geodesic
    '''
    assert p > 0

    @jit
    def nullify(x, v): # closure on `p`

        g = metric(x, bhspin)
        A = v[:p] @ g[:p,:p] @ v[:p]
        b = v[p:] @ g[p:,:p] @ v[:p]
        C = v[p:] @ g[p:,p:] @ v[p:]

        d1, d2 = quadratic(A, b, C)
        S      = jnp.select([d1 > 0, d2 > 0], [d1, d2], jnp.nan)

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
    D  = - (b + bs * jnp.sqrt(dd))
    x1 = D / A
    x2 = C / D
    return jnp.minimum(x1, x2), jnp.maximum(x1, x2)


@jit
def metric(x, bhspin):
    '''
    !@brief Calculates the Kerr-schild metric in Cartesian Co-ordinates.
    @param x 1-D jax array with 4 elements corresponding to {t,x,y,z} which is equivalent to 4-position vector
    @returns a 4 X 4 two dimensional array reprenting the metric

    '''
    eta = jnp.asarray([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    a = bhspin
    aa = a * a
    zz = x[3]*x[3]
    kk = 0.5 * (x[1]*x[1] + x[2]*x[2] + zz - aa)
    rr = jnp.sqrt(kk * kk + aa * zz ) + kk
    r = jnp.sqrt(rr)
    f = (2.0 * rr * r)/(rr * rr + aa * zz)
    l = jnp.array([1, (r * x[1] + a * x[2])/(rr + aa) , (r* x[2] - a * x[1])/(rr + aa) , x[3]/r])
    return eta + f * (l[:,jnp.newaxis] * l[jnp.newaxis,:])


def get_camera_pixel(inclination, distance, radius, angle):

    size = np.size(radius)
    x = jnp.ones(size) * np.cos(angle) * radius  # angle MUST be in radians
    y = jnp.ones(size) * np.sin(angle) * radius
    z = jnp.ones(size) * 0.

    origin_BH  = Image_to_BH(0, 0, 0, inclination, distance)
    temp_coord = perpendicular([x, y])

    init_BH = Image_to_BH(x, y, z, inclination, distance)
    perp_BH = Image_to_BH(temp_coord[0], temp_coord[1], 0 * jnp.ones(1), inclination, distance)

    vec1 = - init_BH.T + origin_BH
    vec2 = perp_BH - init_BH

    k_vec = jnp.cross(vec1, vec2.T)
    s0_x  = jnp.array([0 * jnp.ones(size), init_BH[0].flatten(), init_BH[1].flatten(), init_BH[2].flatten()])
    s0_v  = jnp.array([1 * jnp.ones(size), k_vec.T[0].flatten(), k_vec.T[1].flatten(), k_vec.T[2].flatten()])

    return s0_x, s0_v


def get_initial_grid(i,d,ll,ul,spacing, camera_type):
    '''
    !@brief Defines the initial grid of photons that will be thrown towards the Black hole

    @param i This is the inclination of the observer relative to the black hole in degrees.

    @param d It is the initial distance of the image from the Black hole centre

    @param ll This is the lower limit of the grid

    @param ul This is the upper limit of the grid

    @param equatorial Takes in a boolean value. If True, only photons in the equatorial plane will be initialised with the limits of Impact parameter = ll and ul. If False, defines a grid of photons

    @return s0_x,s0_v Where s0_x is array containing the position 4-vector of all the photons. s0_v contains the velocity 4-vector of all the photons
    '''

    if camera_type.lower() == 'grid':

        grid_list = np.linspace(ll, ul, 2 * spacing + 1)[1::2]

        z = 0 * jnp.ones(len(grid_list)**2)

        x,y = jnp.meshgrid(grid_list,grid_list,indexing = 'ij')

        x = x.flatten()
        y = y.flatten()

        origin_BH = Image_to_BH(0,0,0,i,d)
        temp_coord = perpendicular([x,y])

        init_BH = Image_to_BH(x,y,z,i,d)
        perp_BH = Image_to_BH(temp_coord[0],temp_coord[1],0 * jnp.ones(len(grid_list)**2),i,d)

        vec1 = - init_BH.T + origin_BH
        vec2 = perp_BH - init_BH

        k_vec = np.cross(vec1,vec2.T)
        s0_x = np.array([0 * np.ones(len(grid_list)**2),init_BH[0].flatten(), init_BH[1].flatten(), init_BH[2].flatten()])
        s0_v = np.array([1 * np.ones(len(grid_list)**2),k_vec.T[0].flatten(), k_vec.T[1].flatten(), k_vec.T[2].flatten()])

        return s0_x, s0_v

    elif camera_type.lower() == 'equator':
        grid_list = jnp.linspace(ll, ul, 2 * spacing + 1)[1::2]

        # initialize positions
        s0_x = np.zeros((4, len(grid_list)))
        s0_x[1] = d
        s0_x[2] = grid_list

        # initialize wavevectors
        s0_v = np.ones((4,len(grid_list)))
        s0_v[2] = 0
        s0_v[3] = 0

        return s0_x, s0_v

    else:
        print(f'Unexpected camera type \"{camera_type}\". Please choose either "grid" or "equator"')


def Image_to_BH(x,y,z,i,d):
    i = i * np.pi/180
    x_BH = -y * np.cos(i) + z * np.sin(i) + d * np.sin(i)
    y_BH = x
    z_BH = y * np.sin(i) + z * np.cos(i) + d * np.cos(i)
    return np.array([x_BH,y_BH,z_BH])


def perpendicular( a ) :
    b = np.empty_like(a)
    b[0] = -(0-a[1]) + a[0]
    b[1] = (0-a[0]) + a[1]
    return b

#@jit
def initial_condition_old(s0_x, s0_v, bhspin):
    '''
    !@brief This function returns the correct initial null goedesic conditions for the grid of photons
    '''

    import time
    #print('a', time.time())
    nullify = Nullify(metric, bhspin)

    #print('b', time.time())

    s0 = []
    for i in range(len(s0_x.T)):
        s0.append(np.concatenate([s0_x[:, i], nullify(s0_x[:, i], s0_v[:, i])]))

    #print('c', time.time())

    return np.array(s0)


def initial_condition(s0_x, s0_v, bhspin):
    '''
    !@brief This function returns the correct initial null geodesic conditions for the grid of photons
    '''
    nullify = Nullify(metric, bhspin)

    def concatenate_func(x_col, v_col):
        return jnp.concatenate([x_col, nullify(x_col, v_col)])

    s0 = vmap(concatenate_func, in_axes=(1, 1))(s0_x, s0_v)
    return s0


def geodesic_integrator_new(N, s0, div, tol, bhspin):
    '''
    JAX implementation of the geodesic integrator.
    '''

    states1 = jnp.zeros((N + 1, s0.shape[0], s0.shape[1]))
    final_dt = jnp.zeros((N, s0.shape[0]))
    states1 = states1.at[0, :, :].set(s0)

    #print(states1.shape)
    #print(s0.shape)

    def body_fn(carry, i):
        states, final_dt, idx = carry

        #jaxprint(states[idx - 1], idx)
        #print(states[idx - 1])
        #jaxprint(f"{idx}")  #, final_dt, idx)

        dt = -(radius_cal(states[idx - 1][:, :4], bhspin) - radius_EH(bhspin)) / div
        condition = jnp.logical_or(jnp.abs(dt) * div > 1500, jnp.abs(dt) * div < tol)
        dt = lax.select(condition, jnp.zeros_like(dt), dt)

        new_state = RK4_gen(states[idx - 1], dt, bhspin)  # BOGUE: is idx-1 correct?

        # Store values at fixed index position to maintain shape consistency
        states = states.at[idx].set(new_state)
        final_dt = final_dt.at[idx - 1].set(dt)

        return (states, final_dt, idx + 1), None

    # Run scan with preallocated arrays
    (states1, final_dt, _), _ = lax.scan(body_fn, (states1, final_dt, 1), jnp.arange(N))

    print('c')

    states1 = np.array(states1)
    final_dt = np.array(final_dt)

    # BOGUE: make this use the jnp arrays and be parallelized using jax
    # set every "last" non-zero dt to be zero
    npx = final_dt.shape[1]
    for i in range(npx):
        final_dt[:, i][np.isnan(final_dt[:, i])] = 0
        idx = np.argmax(final_dt[:, i] == 0)
        final_dt[idx-1:, i] = 0.
        states1[idx-1:, i] = states1[idx-1, i]

    print('d')

    return states1, final_dt


def geodesic_integrator_old(N, s0, div, tol, bhspin, use_tqdm=False):
    '''
    !@brief This function gets the Geodesic data and saves it in two arrays, X and V representing position and Velocity
    TODO: fix description (currently inaccurate)
    '''
    states1 = [s0]
    final_dt = []

    print(states1)
    print(s0.shape)

    if use_tqdm:
        iterator = tqdm(range(N))
    else:
        iterator = range(N)

    for i in iterator:

        dt = -(radius_cal_old(states1[-1][:, :4], bhspin) - radius_EH(bhspin))/div

        imp_index = np.where((abs(dt) * div > 1500) | (abs(dt) * div < tol))

        dt[imp_index] = 0.0

        if len(np.where(dt == 0)[0]) == len(dt):
            break

        result1 = RK4_gen(states1[-1], dt, bhspin)
        states1.append(result1)
        final_dt.append(dt)

        #print(states1[:3], result1[:3], dt[:3], i)

        # BOGUE REMOVE (DEBUGGING!)
        #print(dt.shape)
        #print(np.array(final_dt).shape)

        if np.isnan(np.min(result1)) and False:
            print(i)
            for j in range(len(states1[-2])):
                if np.isnan(result1[j]).any():
                    print(j)
                    print('state')
                    print(states1[-2][j])
                    print('dt')
                    print(dt[j])
                    print('result1')
                    print(result1[j])
                    """
            print('states')
            print(states1[-2])
            print('dt')
            print(dt)
            print('result1')
            print(result1)
            """
            break

    S = np.array(states1)
    final_dt = np.array(final_dt)

    # set every "last" non-zero dt to be zero
    npx = final_dt.shape[1]
    for i in range(npx):
        final_dt[:, i][np.isnan(final_dt[:, i])] = 0
        idx = np.argmax(final_dt[:, i] == 0)
        final_dt[idx-1:, i] = 0.
        S[idx-1:, i] = S[idx-1, i]

    return S, final_dt


def radius_cal_old(x, bhspin):
    '''
    !@brief Returns the Spherical Kerr-Schild radius for a point expressed in Cartesian Kerr-Schild coordinates.
    '''
    R = np.sqrt(x[..., 1]**2 + x[..., 2]**2 + x[..., 3]**2)
    return np.sqrt((R**2 - bhspin**2 + np.sqrt((R**2 - bhspin**2)**2 + 4 * bhspin**2 * x[..., 3]**2)) / 2)


def radius_cal(x, bhspin):
    '''
    !@brief Returns the Spherical Kerr-Schild radius for a point expressed in Cartesian Kerr-Schild coordinates.
    '''
    R = jnp.sqrt(x[..., 1]**2 + x[..., 2]**2 + x[..., 3]**2)
    return jnp.sqrt((R**2 - bhspin**2 + jnp.sqrt((R**2 - bhspin**2)**2 + 4 * bhspin**2 * x[..., 3]**2)) / 2)


@jit
def RK4_gen(state1,dt,bhspin):

    val = len(state1)
    ans1 = vectorized_rhs(state1, bhspin)

    k1  = jnp.multiply(dt.reshape(val,1),ans1)

    ans1 = vectorized_rhs(state1 + 0.5 * k1, bhspin)

    k2  = jnp.multiply(dt.reshape(val,1),ans1)

    ans1 = vectorized_rhs(state1 + 0.5 * k2, bhspin)

    k3  = jnp.multiply(dt.reshape(val,1),ans1)

    ans1 = vectorized_rhs(state1 +  k3, bhspin)

    k4  = jnp.multiply(dt.reshape(val,1),ans1)

    new_state1 = state1 + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return new_state1


@jit
def vectorized_rhs(s0, bhspin):
    return vmap(rhs, in_axes=(0, None))(s0,bhspin)


@jit
def rhs(state1, bhspin):
    '''
    !@brief Calculates the RHS of the Geodesic equation
    @param state An n * 8 dimensional array where n = number of photons, 8 = 8 = 4 position coordinates + 4 velocity coordinates

    @returns An n * 8 dimensional array where n = number of photons, 8 = 8 = 4 velocity coordinates + 4 acceleration coordinates
    '''

    x  = state1[:4]
    v  = state1[4:]

    ig = imetric(x, bhspin)
    jg = jit(jacfwd(metric))(x, bhspin)

    a  = ig @ (- (jg @ v) @ v + 0.5 * v @ (v @ jg))

    return jnp.concatenate([v,a]) #a - a[0] * v


@jit
def imetric(x, bhspin):
    '''
    !@brief Used by rhs()
    @param x The 4 vector position
    @returns The inverse of the metric at position x.

    '''
    return inv(metric(x, bhspin))


def radius_EH(a_spin):
    return 1 + np.sqrt(1-a_spin**2)


def select_photons_integrator(N_time_steps, inc, angle, radius, bhspin, distance = 1000):
    s0_x, s0_v = get_camera_pixel(inc,distance,radius,angle)
    init_one   = initial_condition(s0_x, s0_v, bhspin)
    S,final_dt = geodesic_integrator(N_time_steps,init_one,40,1e-2,bhspin)
    r = radius_cal(S, bhspin)
    return(np.nanmin(r, axis=0)) # returns minimum distance from BH


def bisection_shadow_par(bhspin, inc, num_angles, N_time_steps=2000, uncertainty_allowed=0.001, max_it=40):
    inner  = np.zeros(num_angles) + 0.5
    outer  = np.zeros(num_angles) + 10
    error  = outer - inner
    angles = np.arange(num_angles)/num_angles*2.*np.pi

    reh = 1. + np.sqrt(1. - bhspin*bhspin)
    reh = reh + 0.05

    counter = 0
    while (np.max(error) > uncertainty_allowed) and (counter < max_it):
        final_mid       = select_photons_integrator(N_time_steps, inc, angles, (outer-inner)/2+inner, bhspin)
        fell            = np.where(final_mid < reh)
        got_away        = np.where(final_mid > reh)
        inner[fell]     = (outer[fell]-inner[fell])/2+inner[fell]
        outer[got_away] = (outer[got_away]-inner[got_away])/2+inner[got_away]
        error = outer - inner
        counter+=1

    angles = np.append(angles, angles[0])
    inner  = np.append(inner,   inner[0])

    return(angles, inner)

