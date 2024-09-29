
import numpy as np

from jax import numpy as jnp
from jax import jit, jacfwd, vmap
from jax.numpy import dot
from jax.numpy.linalg import inv
##### This will tell whether we are using CPU or GPU #########
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

def initialize_geodesics_at_camera(i,d,ll,ul,spacing,TYPE, bhspin):
    '''
    @ Brief This function generates an initial grid of photons that are then used for the geodesic integrator.

    @ dependencies: This function depends on the following functions,
    Nullify(metric, p=1)
        quadratic(A, b, C)
        metric(x)

    initial_grid(i,d,ll,ul,spacing,Type)
        meshgrid
        Image_to_BH(x,y,z,i,d)
        perpendicular( a )

    init_cond(s0_x,s0_v)
        Nullify(metric, p=1)
            quadratic(A, b, C)
            metric(x)
    '''
    '''
    Initilializes the image plane.

    i - inclination (in degrees)
    d - ditance of the image from the black hole (in code units)
    ll - lower limit of the image plane
    ul - upper limit of the image plane
    spacing - number of photons between ul and ll
    TYPE - it's either GRID (for 3-D visualization) or Equator (for photons along the equatorial cross-section)
    '''

    s0_x, s0_v = initial_grid(i,d,ll,ul,spacing,TYPE)

    return init_cond(s0_x, s0_v, bhspin)


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


def initial_grid(i,d,ll,ul,spacing,Type):
    '''
    !@brief Defines the initial grid of photons that will be thrown towards the Black hole

    @param i This is the inclination of the observer relative to the black hole in degrees.

    @param d It is the initial distance of the image from the Black hole centre

    @param ll This is the lower limit of the grid

    @param ul This is the upper limit of the grid

    @param equatorial Takes in a boolean value. If True, only photons in the equatorial plane will be initialised with the limits of Impact parameter = ll and ul. If False, defines a grid of photons

    @return s0_x,s0_v Where s0_x is array containing the position 4-vector of all the photons. s0_v contains the velocity 4-vector of all the photons
    '''

    if Type == 'Grid':

        grid_list = np.linspace(ll,ul,2 * spacing + 1)[1::2]

        z = 0 * jnp.ones(len(grid_list)**2)

        x,y = jnp.meshgrid(grid_list,grid_list,indexing = 'ij')

        x = x.flatten()
        y = y.flatten()

        origin_BH = Image_to_BH(0,0,0,i,d)
        temp_coord = perpendicular([x,y])

        init_BH = Image_to_BH(x,y,z,i,d)
        perp_BH = Image_to_BH(temp_coord[0],temp_coord[1],0 * jnp.ones(len(grid_list)**2),i,d)

        vec1 = - (init_BH.T - origin_BH)
        vec2 = (perp_BH - init_BH)

        k_vec = np.cross(vec1,vec2.T)
        s0_x = np.array([0 * np.ones(len(grid_list)**2),init_BH[0].flatten(), init_BH[1].flatten(), init_BH[2].flatten()])
        s0_v = np.array([1 * np.ones(len(grid_list)**2),k_vec.T[0].flatten(), k_vec.T[1].flatten(), k_vec.T[2].flatten()])

        return s0_x,s0_v

    elif Type == 'Equator':
        grid_list = jnp.arange(ll,ul,spacing)

        s0 = jnp.zeros(8)

        s0_x = np.zeros((4,len(grid_list)))

        s0_x[1,:] = d
        s0_x[2,:] = grid_list

        s0_v = np.ones((4,len(grid_list)))

        s0_v[2,:] = 0
        s0_v[3,:] = 0

        return s0_x,s0_v

    else:
        pass


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


def init_cond(s0_x, s0_v, bhspin):
    '''
    !@breif This function returns the correct initial null goedesic conditions for the grid of photons
    '''

    nullify = Nullify(metric, bhspin)

    s0 = []
    for i in range(len(s0_x.T)):
        v = nullify(s0_x[:,i],s0_v[:,i])
        s0.append(np.concatenate([s0_x[:,i],v]))

    s0 = np.array(s0)
    return s0


