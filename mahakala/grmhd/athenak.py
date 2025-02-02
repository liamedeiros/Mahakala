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
import h5py

from jax.scipy.interpolate import RegularGridInterpolator as jaxRegularGridInterpolator

from tqdm import tqdm

from jax import lax
from jax import numpy as jnp
from jax import jit, vmap

import time

from mahakala.geodesics import metric, imetric


KB = 1.3807e-16
CL = 2.99792458e10
ME = 9.1094e-28
EC = 4.8032e-10
HPL = 6.6261e-27
GNEWT = 6.6743e-8


@jit
def vec_metric(X, bhspin):
    return vmap(metric, in_axes=(0, None))(X, bhspin)


@jit
def vec_imetric(X, bhspin):
    return vmap(imetric, in_axes=(0, None))(X, bhspin)


class AthenakFluidModel:

    def __init__(self, grmhd_filename, bhspin):
        self.load_athenak_meshblocks(grmhd_filename)
        self.bhspin = bhspin

    def get_index_for_primitive_by_name(self, prim):
        """
        Return which index in the meshblock array corresponds to the desired
        primitive variable. Return -1 when the desired primitive variable is
        not found.
        """
        prim = prim.lower().strip()
        var_names = [v.lower().strip() for v in self.variable_names]
        if prim in var_names:
            return var_names.index(prim)
        return -1

    def get_extended(self, arr):
        """
        Extend a one-dimensional array by one element on each side and fill
        the new elements with linear extrapolation.
        """
        arr2 = np.zeros(arr.size + 2)
        arr2[1:-1] = arr
        dx = arr[1] - arr[0]
        arr2[0] = arr2[1] - dx
        arr2[-1] = arr2[-2] + dx
        return arr2

    def get_all_vals_unique(self, xs):
        """
        Get all unique values in a list of arrays.
        """
        all_xs = []
        for x in xs:
            all_xs += list(x)
        return np.array(sorted(list(set(all_xs))))

    def load_athenak_meshblocks(self, grmhd_filename):
        """
        Load meshblocks from an athenak athdf file. The meshblocks are stored
        with ghost zones in a 5-dimensional array self.all_meshblocks[5] with
        the shape (nmb, nprim, nk+2, nj+2, ni+2), where
            - nmb is the number of meshblocks,
            - nprim is the number of primitive variables,
            - nk, nj, ni are the numbers of zones in each direction.

        The names of the primitives are saved in self.variable_names.
        """

        with h5py.File(grmhd_filename, 'r') as hfp:

            x1v = np.array(hfp['x1v'])
            x2v = np.array(hfp['x2v'])
            x3v = np.array(hfp['x3v'])
            x1f = np.array(hfp['x1f'])
            x2f = np.array(hfp['x2f'])
            x3f = np.array(hfp['x3f'])
            uov = np.array(hfp['uov'])

            B = np.array(hfp['B'])
            LogicalLocations = np.array(hfp['LogicalLocations'])
            Levels = np.array(hfp['Levels'])
            self.variable_names = np.array([n.decode('utf-8') for n in hfp.attrs['VariableNames']])
            hfp.close()

            # uov[0] -> dens
            # uov[1] -> velx
            # uov[2] -> vely
            # uov[3] -> velz
            # uov[4] -> eint
            # B[0] -> bcc1
            # B[1] -> bcc2
            # B[2] -> bcc3

            min_level = int(Levels.min())
            max_level = int(Levels.max())

            max_l1_i = LogicalLocations[Levels == min_level][:, 0].max()
            max_l1_j = LogicalLocations[Levels == min_level][:, 1].max()
            max_l1_k = LogicalLocations[Levels == min_level][:, 2].max()

            nprim, nmb, nmbk, nmbj, nmbi = uov.shape

            nprim_all = 8

            mb_index_map = {}
            for mb in range(nmb):
                tlevel = Levels[mb]
                ti, tj, tk = LogicalLocations[mb]
                key = tlevel, ti, tj, tk
                mb_index_map[key] = mb


            all_x1s = self.get_all_vals_unique(x1v)
            all_x2s = self.get_all_vals_unique(x2v)
            all_x3s = self.get_all_vals_unique(x3v)

            extrema = np.abs(np.array([all_x1s.min(), all_x1s.max(), all_x2s.min(),
                                    all_x2s.max(), all_x3s.min(), all_x3s.max()]))

            all_meshblocks = []
            for mbi in tqdm(mb_index_map.values()):

                # get edges for grid interpolator
                x1e = self.get_extended(x1v[mbi])
                x2e = self.get_extended(x2v[mbi])
                x3e = self.get_extended(x3v[mbi])

                # get meshblock key information
                tlevel = Levels[mbi]
                ti, tj, tk = LogicalLocations[mbi]
                key = tlevel, ti, tj, tk

                # fill the center of the interpolating meshblock object
                new_meshblock = np.zeros((nprim_all, nmbi+2, nmbj+2, nmbk+2))
                new_meshblock[:nprim, 1:-1, 1:-1, 1:-1] = uov[:, mbi]
                new_meshblock[nprim:, 1:-1, 1:-1, 1:-1] = B[:, mbi]

                # populate boundaries of meshblock for interpolation
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:

                            # skip center block
                            if di == 0 and dj == 0 and dk == 0:
                                continue

                            mb_info = [tlevel, ti, tj, tk, di, dj, dk, nprim, nmbi, nmbj, nmbk]
                            new_meshblock = self.get_new_meshblock_boundary(*mb_info, mb_index_map,
                                                                            new_meshblock, uov, B)

                all_meshblocks.append(new_meshblock)

            self.mb_index_map = mb_index_map

            self.all_meshblocks = np.array(all_meshblocks)

            self.x1v = x1v
            self.x2v = x2v
            self.x3v = x3v

            self.x1f = x1f
            self.x2f = x2f
            self.x3f = x3f

            self.Levels = Levels
            self.LogicalLocations = LogicalLocations

            self.nprim_all = nprim_all


    def get_key_for_level(self, c_level, n_level, ti, tj, tk, di, dj, dk):
        '''
        WARNING: This fails if abs(c_level-n_level) > 1.
        '''

        nti = ti
        ntj = tj
        ntk = tk

        if c_level == n_level:
            return ti+di, tj+dj, tk+dk

        while c_level < n_level:
            c_level += 1
            nti = 2 * (nti + di)
            ntj = 2 * (ntj + dj)
            ntk = 2 * (ntk + dk)
            if c_level == n_level:
                return c_level, nti, ntj, ntk
            break

        while c_level > n_level:
            c_level -= 1
            nti = (nti + di) // 2
            ntj = (ntj + dj) // 2
            ntk = (ntk + dk) // 2
            if c_level == n_level:
                return c_level, nti, ntj, ntk
            break

        raise Exception("Unable to compute key for meshblock level")

    def get_slice_source(self, ntot, oddity):
        if oddity == 1:
            return slice(ntot//2, ntot)
        else:
            return slice(0, ntot//2)

    def get_01_source(self, v, ntot, oddity):
        if oddity == 1:
            if v == 0:
                return ntot//2
            return -1
        else:
            if v == 0:
                return 0
            return ntot // 2 - 1

    def get_new_meshblock_boundary(self, tlevel, ti, tj, tk, di, dj, dk,
                                   nprim, nmbi, nmbj, nmbk, mb_index_map,
                                   new_meshblock, uov, B):

        # first see if we can stay on the same level
        trial_key = tlevel, ti+di, tj+dj, tk+dk
        if trial_key in mb_index_map:

            nmb = mb_index_map[trial_key]

            src_i = 0 if di == 1 else (-1 if di == -1 else slice(0, nmbi))
            src_j = 0 if dj == 1 else (-1 if dj == -1 else slice(0, nmbj))
            src_k = 0 if dk == 1 else (-1 if dk == -1 else slice(0, nmbk))

            tgt_i = -1 if di == 1 else (0 if di == -1 else slice(1, nmbi+1))
            tgt_j = -1 if dj == 1 else (0 if dj == -1 else slice(1, nmbj+1))
            tgt_k = -1 if dk == 1 else (0 if dk == -1 else slice(1, nmbk+1))

            new_meshblock[:nprim, tgt_k, tgt_j, tgt_i] = uov[:, nmb, src_k, src_j, src_i]
            new_meshblock[nprim:, tgt_k, tgt_j, tgt_i] = B[:, nmb, src_k, src_j, src_i]

            return new_meshblock

        # then see if we can go up one level
        trial_key = self.get_key_for_level(tlevel, tlevel-1, ti, tj, tk, di, dj, dk)
        if trial_key in mb_index_map:

            nmb = mb_index_map[trial_key]

            _, newi, newj, newk = trial_key

            oddi = (ti+di) % 2
            oddj = (tj+dj) % 2
            oddk = (tk+dk) % 2

            source_i = self.get_01_source(0, nmbi, oddi) if di == 1 else (self.get_01_source(-1, nmbi, oddi) if di == -1 else self.get_slice_source(nmbi, oddi))
            source_j = self.get_01_source(0, nmbj, oddj) if dj == 1 else (self.get_01_source(-1, nmbj, oddj) if dj == -1 else self.get_slice_source(nmbj, oddj))
            source_k = self.get_01_source(0, nmbk, oddk) if dk == 1 else (self.get_01_source(-1, nmbk, oddk) if dk == -1 else self.get_slice_source(nmbk, oddk))

            for slc_i in range(2):
                for slc_j in range(2):
                    for slc_k in range(2):
                        target_i = -1 if di == 1 else (0 if di == -1 else slice(1+slc_i, nmbi+slc_i+1, 2))
                        target_j = -1 if dj == 1 else (0 if dj == -1 else slice(1+slc_j, nmbj+slc_j+1, 2))
                        target_k = -1 if dk == 1 else (0 if dk == -1 else slice(1+slc_k, nmbk+slc_k+1, 2))

                        new_meshblock[:nprim, target_k, target_j, target_i] = uov[:, nmb, source_k, source_j, source_i]
                        new_meshblock[nprim:, target_k, target_j, target_i] = B[:, nmb, source_k, source_j, source_i]

            return new_meshblock

        # figure out how many slices we have (corner vs. edge vs. face)
        num_slices = 3
        if di in [-1, 1]:
            num_slices -= 1
        if dj in [-1, 1]:
            num_slices -= 1
        if dk in [-1, 1]:
            num_slices -= 1

        # finally see if we can go down one level
        trial_key = self.get_key_for_level(tlevel, tlevel+1, ti, tj, tk, di, dj, dk)
        if trial_key in mb_index_map:

            # always need this, since we need to deal with offsets
            newlevel, newi, newj, newk = trial_key

            # handle corners
            if num_slices == 0:

                source_i = 0
                source_j = 0
                source_k = 0

                # 2x as many meshblocks at this level, adjust
                if di == -1:
                    newi += 1
                    source_i = nmbi - 2
                if dj == -1:
                    newj += 1
                    source_j = nmbj - 2
                if dk == -1:
                    newk += 1
                    source_k = nmbk - 2

                trial_key = newlevel, newi, newj, newk
                nmb = mb_index_map[trial_key]

                target_i = -1 if di == 1 else 0
                target_j = -1 if dj == 1 else 0
                target_k = -1 if dk == 1 else 0

                for v1 in range(2):
                    for v2 in range(2):
                        for v3 in range(2):

                            contribution = uov[:, nmb, source_k + v3, source_j + v2, source_i + v1]
                            new_meshblock[:nprim, target_k, target_j, target_i] += 0.125 * contribution

                            contribution = B[:, nmb, source_k + v3, source_j + v2, source_i + v1]
                            new_meshblock[nprim:, target_k, target_j, target_i] += 0.125 * contribution

                return new_meshblock

            # now handle edges
            elif num_slices == 1:

                source_i = 0
                source_j = 0
                source_k = 0

                # 2x as many meshblocks at this level, adjust
                if di == -1:
                    newi += 1
                    source_i = nmbi - 2
                if dj == -1:
                    newj += 1
                    source_j = nmbj - 2
                if dk == -1:
                    newk += 1
                    source_k = nmbk - 2

                for edge_pos in range(2):

                    slc_i = 0
                    slc_j = 0
                    slc_k = 0

                    # realign targets for corners
                    if di in [-1, 1]:
                        target_i = -1 if di == 1 else 0
                    else:
                        target_i = slice(1+edge_pos*nmbi//2, 1+(1+edge_pos)*nmbi//2)
                        slc_i = edge_pos
                        source_i = slice(1)

                    if dj in [-1, 1]:
                        target_j = -1 if dj == 1 else 0
                    else:
                        target_j = slice(1+edge_pos*nmbj//2, 1+(1+edge_pos)*nmbj//2)
                        slc_j = edge_pos
                        source_j = slice(1)

                    if dk in [-1, 1]:
                        target_k = -1 if dk == 1 else 0
                    else:
                        target_k = slice(1+edge_pos*nmbk//2, 1+(1+edge_pos)*nmbk//2)
                        slc_k = edge_pos
                        source_k = slice(1)

                    trial_key = newlevel, newi+slc_i, newj+slc_j, newk+slc_k
                    nmb = mb_index_map[trial_key]

                    for v1 in range(2):
                        for v2 in range(2):
                            for v3 in range(2):

                                copy_source_i = source_i
                                if isinstance(source_i, int):
                                    copy_source_i += v1
                                else:
                                    source_i = slice(v1, nmbi+v1, 2)

                                copy_source_j = source_j
                                if isinstance(source_j, int):
                                    copy_source_j += v2
                                else:
                                    source_j = slice(v2, nmbj+v2, 2)

                                copy_source_k = source_k
                                if isinstance(source_k, int):
                                    copy_source_k += v3
                                else:
                                    source_k = slice(v3, nmbk+v3, 2)

                                contribution = uov[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                                new_meshblock[:nprim, target_k, target_j, target_i] += contribution / 8.

                                contribution = B[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                                new_meshblock[nprim:, target_k, target_j, target_i] += contribution / 8.

                return new_meshblock

            # now handle faces
            elif num_slices == 2:

                source_i = 0
                source_j = 0
                source_k = 0

                # 2x as many meshblocks at this level, adjust
                if di == -1:
                    newi += 1
                    source_i = nmbi - 2
                if dj == -1:
                    newj += 1
                    source_j = nmbj - 2
                if dk == -1:
                    newk += 1
                    source_k = nmbk - 2

                for edge_pos_1 in range(2):
                    for edge_pos_2 in range(2):

                        slc_i = 0
                        slc_j = 0
                        slc_k = 0

                        if di in [-1, 1]:
                            edge_pos_j = edge_pos_1
                            edge_pos_k = edge_pos_2
                        elif dj in [-1, 1]:
                            edge_pos_i = edge_pos_1
                            edge_pos_k = edge_pos_2
                        else:
                            edge_pos_i = edge_pos_1
                            edge_pos_j = edge_pos_2

                        # realign targets for corners
                        if di in [-1, 1]:
                            target_i = -1 if di == 1 else 0
                        else:
                            target_i = slice(1+edge_pos_i*nmbi//2, 1+(1+edge_pos_i)*nmbi//2)
                            slc_i = edge_pos_i
                            source_i = slice(1)

                        if dj in [-1, 1]:
                            target_j = -1 if dj == 1 else 0
                        else:
                            target_j = slice(1+edge_pos_j*nmbj//2, 1+(1+edge_pos_j)*nmbj//2)
                            slc_j = edge_pos_j
                            source_j = slice(1)

                        if dk in [-1, 1]:
                            target_k = -1 if dk == 1 else 0
                        else:
                            target_k = slice(1+edge_pos_k*nmbk//2, 1+(1+edge_pos_k)*nmbk//2)
                            slc_k = edge_pos_k
                            source_k = slice(1)

                        trial_key = newlevel, newi+slc_i, newj+slc_j, newk+slc_k
                        nmb = mb_index_map[trial_key]

                        for v1 in range(2):
                            for v2 in range(2):
                                for v3 in range(2):

                                    copy_source_i = source_i
                                    if type(source_i) == int:
                                        copy_source_i += v1
                                    else:
                                        copy_source_i = slice(v1, nmbi+v1, 2)

                                    copy_source_j = source_j
                                    if type(source_j) == int:
                                        copy_source_j += v2
                                    else:
                                        copy_source_j = slice(v2, nmbj+v2, 2)

                                    copy_source_k = source_k
                                    if type(source_k) == int:
                                        copy_source_k += v3
                                    else:
                                        copy_source_k = slice(v3, nmbk+v3, 2)

                                    contribution = uov[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                                    new_meshblock[:nprim, target_k, target_j, target_i] += contribution / 8.

                                    contribution = B[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                                    new_meshblock[nprim:, target_k, target_j, target_i] += contribution / 8.

                return new_meshblock

        # we should only have trouble populating boundaries when we extend
        # beyond the domain. if instead we end up printing a message here,
        # there's something odd afoot.
        tvi = ti + di
        tvj = tj + dj
        tvk = tk + dk
        if tvi >= 0 and tvj >= 0 and tvk >= 0:
            pass

        return new_meshblock

    def get_prims_from_geodesics(self, S, interp_method='linear'):
        """
        TODO: documentation
        
        TODO: ensure that meshblock_indices == -1  =>  prims = 0


        """
        
        nsteps, npx, _ = S.shape
        nmb = len(self.mb_index_map.values())

        # get which meshblock each geodesic point is in
        mb_indices = np.ones((nsteps, npx), dtype=int) * -1

        ext_x1 = np.array([[self.x1f[mbi][0], self.x1f[mbi][-1]] for mbi in range(nmb)])
        ext_x2 = np.array([[self.x2f[mbi][0], self.x2f[mbi][-1]] for mbi in range(nmb)])
        ext_x3 = np.array([[self.x3f[mbi][0], self.x3f[mbi][-1]] for mbi in range(nmb)])

        t0 = time.time()

        for mbi in tqdm(range(nmb)):
            mb_mask = (ext_x1[mbi][0] < S[..., 1]) & (S[..., 1] <= ext_x1[mbi][1])
            mb_mask &= (ext_x2[mbi][0] < S[..., 2]) & (S[..., 2] <= ext_x2[mbi][1])
            mb_mask &= (ext_x3[mbi][0] < S[..., 3]) & (S[..., 3] <= ext_x3[mbi][1])
            mb_indices[mb_mask] = mbi

        mb_indices = jnp.array(mb_indices)

        t1 = time.time()
        print(t1 - t0)

        # get the primitive data
        x1_left = jnp.array([self.x1v[mbi][0] for mbi in range(nmb)])
        x2_left = jnp.array([self.x2v[mbi][0] for mbi in range(nmb)])
        x3_left = jnp.array([self.x3v[mbi][0] for mbi in range(nmb)])
        dx1 = jnp.array([self.x1v[mbi][1] - self.x1v[mbi][0] for mbi in range(nmb)])
        dx2 = jnp.array([self.x2v[mbi][1] - self.x2v[mbi][0] for mbi in range(nmb)])
        dx3 = jnp.array([self.x3v[mbi][1] - self.x3v[mbi][0] for mbi in range(nmb)])

        meshblock_data = jnp.array(self.all_meshblocks)

        # return the primitives evaluated at the points given by S[i, :, :4]
        def body_fn(_, i):

            # consider the ith step
            S0 = S[i]

            # get positions and offsets in meshblocks
            x1_indices = S0[:, 1] - x1_left[mb_indices[i]] + dx1[mb_indices[i]]
            x2_indices = S0[:, 2] - x2_left[mb_indices[i]] + dx2[mb_indices[i]]
            x3_indices = S0[:, 3] - x3_left[mb_indices[i]] + dx3[mb_indices[i]]

            x1_delta = x1_indices / dx1[mb_indices[i]]
            x2_delta = x2_indices / dx2[mb_indices[i]]
            x3_delta = x3_indices / dx3[mb_indices[i]]

            # add one to correct for ghost zones
            x1_indices = jnp.array(x1_indices // dx1[mb_indices[i]], dtype=int)
            x2_indices = jnp.array(x2_indices // dx2[mb_indices[i]], dtype=int)
            x3_indices = jnp.array(x3_indices // dx3[mb_indices[i]], dtype=int)

            x1_delta = jnp.array(x1_delta % 1.)
            x2_delta = jnp.array(x2_delta % 1.)
            x3_delta = jnp.array(x3_delta % 1.)

            # "manual" linear interpolation
            # naming convention is x3, x2, x1 with (a, b) -> (0, +1)
            daaa = meshblock_data[mb_indices[i], :, x3_indices, x2_indices, x1_indices]
            daab = meshblock_data[mb_indices[i], :, x3_indices, x2_indices, x1_indices + 1]
            daba = meshblock_data[mb_indices[i], :, x3_indices, x2_indices + 1, x1_indices]
            dabb = meshblock_data[mb_indices[i], :, x3_indices, x2_indices + 1, x1_indices + 1]
            dbaa = meshblock_data[mb_indices[i], :, x3_indices + 1, x2_indices, x1_indices]
            dbab = meshblock_data[mb_indices[i], :, x3_indices + 1, x2_indices, x1_indices + 1]
            dbba = meshblock_data[mb_indices[i], :, x3_indices + 1, x2_indices + 1, x1_indices]
            dbbb = meshblock_data[mb_indices[i], :, x3_indices + 1, x2_indices + 1, x1_indices + 1]

            daa = daaa + (daab - daaa) * x1_delta[:, None]
            dab = daba + (dabb - daba) * x1_delta[:, None]
            dba = dbaa + (dbab - dbaa) * x1_delta[:, None]
            dbb = dbba + (dbbb - dbba) * x1_delta[:, None]
            da = daa + (dab - daa) * x2_delta[:, None]
            db = dba + (dbb - dba) * x2_delta[:, None]
            prims = da + (db - da) * x3_delta[:, None]

            # update prim array to be zero wherever the meshblock index is -1
            condition = mb_indices[i] == -1
            condition = jnp.broadcast_to(condition[:, None], prims.shape)
            prims = lax.select(condition, jnp.zeros_like(prims), prims)

            return _, prims

        _, prims = lax.scan(body_fn, None, jnp.arange(nsteps))

        t2 = time.time()
        print(t2 - t1)

        primitive_data = dict(
            dens=prims[..., self.get_index_for_primitive_by_name('dens')],
            u=prims[..., self.get_index_for_primitive_by_name('eint')],
            U1=prims[..., self.get_index_for_primitive_by_name('velx')],
            U2=prims[..., self.get_index_for_primitive_by_name('vely')],
            U3=prims[..., self.get_index_for_primitive_by_name('velz')],
            B1=prims[..., self.get_index_for_primitive_by_name('bcc1')],
            B2=prims[..., self.get_index_for_primitive_by_name('bcc2')],
            B3=prims[..., self.get_index_for_primitive_by_name('bcc3')],
        )

        return primitive_data

    def compute_tensorial(self, S, primitive_data):
        """
        TODO: documentation
        """

        nsteps, npx, _ = S.shape

        UuUu = np.zeros((nsteps, npx))

        total_u = np.array([primitive_data['U1'], primitive_data['U2'], primitive_data['U3']])
        total_u = np.transpose(total_u, (1, 2, 0))

        for i in tqdm(range(nsteps)):
            UuUu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin)[:,1:,1:] @ total_u[i,:,:].reshape(npx,3,1)).reshape(npx,1,3) @ total_u[i,:,:].reshape(npx,3,1)).reshape(npx)
        del total_u

        GAMMA = np.sqrt(1 + UuUu)
        del UuUu

        final_M = np.zeros((nsteps, npx, 4))
        for i in tqdm(range(nsteps)):
            final_M[i,:,:] = vec_imetric(S[i,:,:4], self.bhspin)[:,0,:]

        u0_data = (GAMMA/pow(-final_M[:,:,0], -1/2))
        u1_data = (primitive_data['U1'] - (final_M[:,:,1] * GAMMA * pow(-final_M[:,:,0],-1/2)))
        u2_data = (primitive_data['U2'] - (final_M[:,:,2] * GAMMA * pow(-final_M[:,:,0],-1/2)))
        u3_data = (primitive_data['U3'] - (final_M[:,:,3] * GAMMA * pow(-final_M[:,:,0],-1/2)))

        del final_M

        #######################
        # Now, for the magnetic field components

        BuUu = np.zeros((nsteps, npx))

        total_B = np.array([primitive_data['B1'], primitive_data['B2'], primitive_data['B3']])
        total_B = np.transpose(total_B, (1, 2, 0))

        total_u = np.array([u0_data, u1_data, u2_data, u3_data])
        total_u = np.transpose(total_u, (1, 2, 0))

        for i in tqdm(range(nsteps)):
            BuUu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin) @ total_u[i,:,:].reshape(npx,4,1)).reshape(npx,1,4)[:,:,1:] @ total_B[i,:,:].reshape(npx,3,1)).reshape(npx)

        del total_B

        B0_data = BuUu
        del BuUu
        B1_data = 1/u0_data * (primitive_data['B1'] + B0_data * u1_data)
        B2_data = 1/u0_data * (primitive_data['B2'] + B0_data * u2_data)
        B3_data = 1/u0_data * (primitive_data['B3'] + B0_data * u3_data)

        KuUu = np.zeros((nsteps, npx))

        for i in tqdm(range(nsteps)):
            KuUu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin) @ total_u[i,:,:].reshape(npx,4,1)).reshape(npx,1,4) @ S[i,:,4:].reshape(npx,4,1)).reshape(npx)

        total_B = np.array([B0_data, B1_data, B2_data, B3_data])
        tot_B = np.transpose(total_B, (1,2,0))

        KuBu = np.zeros((nsteps, npx))

        for i in tqdm(range(nsteps)):
            KuBu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin) @ tot_B[i,:,:].reshape(npx,4,1)).reshape(npx,1,4) @ S[i,:,4:].reshape(npx,4,1)).reshape(npx)

        BuBu = np.zeros((nsteps, npx))

        for i in tqdm(range(nsteps)):
            BuBu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin) @ tot_B[i,:,:].reshape(npx,4,1)).reshape(npx,1,4) @ tot_B[i,:,:].reshape(npx,4,1)).reshape(npx)

        del tot_B
        del total_B

        ## TODO remove UdotU
        UuUu = np.zeros((nsteps, npx))

        del total_u
        total_u = np.array([u0_data, u1_data, u2_data, u3_data])
        tot_u = np.transpose(total_u, (1, 2, 0))

        for i in tqdm(range(nsteps)):
            UuUu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin) @ tot_u[i,:,:].reshape(npx,4,1)).reshape(npx,1,4) @ tot_u[i,:,:].reshape(npx,4,1)).reshape(npx)

        def compute_pitch_angle(KuUu, KuBu, BuBu):

            angle = KuBu/(np.abs(KuUu) * np.sqrt(BuBu))

            index = np.where(BuBu == 0)
            angle[index[0],index[1]] = np.cos(np.pi/2)

            index = np.where(abs(angle) > 1.0)
            angle[index[0],index[1]] = angle[index[0],index[1]]/abs(angle[index[0],index[1]])

            return np.arccos(angle)

        observer_angle = compute_pitch_angle(KuUu,KuBu,BuBu)

        tensorial_data = dict(
            ucon = np.array([u0_data, u1_data, u2_data, u3_data]),
            bcon = np.array([B0_data, B1_data, B2_data, B3_data]),
            pitch_angle = observer_angle,
            udotu = UuUu,
            kdotu = KuUu,
            kdotb = KuBu,
            bdotb = BuBu
        )

        return tensorial_data
