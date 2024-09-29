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
from jax import jit, jacfwd, vmap
from jax.numpy import dot
from jax.numpy.linalg import inv
##### This will tell whether we are using CPU or GPU #########
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


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

    z = jnp.ones(1)
    x, y = jnp.meshgrid([radius], [angle], indexing='ij')

    x = x.flatten()
    y = y.flatten()

    origin_BH = Image_to_BH(0, 0, 0, inclination, distance)
    temp_coord = perpendicular([x, y])

    init_BH = Image_to_BH(x, y, z, inclination, distance)
    perp_BH = Image_to_BH(temp_coord[0], temp_coord[1], 0 * jnp.ones(1), inclination, distance)

    vec1 = - init_BH.T + origin_BH
    vec2 = perp_BH - init_BH

    k_vec = jnp.cross(vec1, vec2.T)
    s0_x = jnp.array([0 * jnp.ones(1), init_BH[0].flatten(), init_BH[1].flatten(), init_BH[2].flatten()])
    s0_v = jnp.array([1 * jnp.ones(1), k_vec.T[0].flatten(), k_vec.T[1].flatten(), k_vec.T[2].flatten()])

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

        return s0_x,s0_v

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


def initial_condition(s0_x, s0_v, bhspin):
    '''
    !@breif This function returns the correct initial null goedesic conditions for the grid of photons
    '''

    nullify = Nullify(metric, bhspin)

    s0 = []
    for i in range(len(s0_x.T)):
        v = nullify(s0_x[:, i], s0_v[:, i])
        s0.append(np.concatenate([s0_x[:, i], v]))

    return np.array(s0)


def geodesic_integrator(N, s0,div,tol, bhspin):
    '''
    !@brief This function gets the Geodesic data and saves it in two arrays, X and V representing position and Velocity
    '''
    states1 = [s0]
    final_dt = []

    for i in tqdm(range(N)):
        dt = -(radius_cal(states1[-1][:, :4], bhspin) - radius_EH(bhspin))/div

        imp_index = np.where((abs(dt) * div > 1500) | (abs(dt) < tol))

        dt[imp_index] = 0.0

        if len(np.where(dt == 0)[0]) == len(dt):
            break

        result1 = RK4_gen(states1[-1],dt,bhspin)
        states1.append(result1)
        final_dt.append(dt)

    S = np.array(states1)

    return S,np.array(final_dt)


def radius_cal(x, bhspin):

    R = np.sqrt(x[:,1]**2 + x[:,2]**2 + x[:,3]**2)
    r = np.sqrt((R**2 - bhspin**2  + np.sqrt((R**2 - bhspin**2)**2 + 4 * bhspin**2 * x[:,3]**2)) / 2)
    return r


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
    '''
    !@brief This uses Jax vmap. It has the familiar semantics of mapping a function along array axes, but instead of
    keeping the loop on the outside, it pushes the loop down into a function’s primitive operations for better
    performance. When composed with jit(), it can be just as fast as adding the batch dimensions by hand.
    '''
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

