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

from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm

from jax import numpy as jnp
from jax import jit, jacfwd, vmap
from jax.numpy import dot
from jax.numpy.linalg import inv

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


    def map_prim_to_prim(self, remapped, nprm, variable_names, fluid_params):
        '''
        TODO: In future check that we are accurately dealing with mapping of input
        fluid variables to standard RHO, UU, U1, U2, U3, B1, B2, B3.
        '''
        if nprm == 0:
            return 0, remapped
        if nprm in [1, 2, 3]:
            return nprm + 1, remapped
        if nprm == 4:
            return 1, remapped
        return nprm, remapped


    def get_extended(self, arr):
        arr2 = np.zeros(arr.size + 2)
        arr2[1:-1] = arr
        dx = arr[1] - arr[0]
        arr2[0] = arr2[1] - dx
        arr2[-1] = arr2[-2] + dx
        return arr2


    def get_all_vals_unique(self, xs):
        all_xs = []
        for x in xs:
            all_xs += list(x)
        return np.array(sorted(list(set(all_xs))))


    def load_athenak_meshblocks(self, grmhd_filename):

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
            self.variable_names = np.array(hfp.attrs['VariableNames'])
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


    def get_new_meshblock_boundary(self, tlevel, ti, tj, tk, di, dj, dk, nprim, nmbi, nmbj, nmbk, mb_index_map, new_meshblock, uov, B):

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
                                if type(source_i) == int:
                                    copy_source_i += v1
                                else:
                                    source_i = slice(v1, nmbi+v1, 2)

                                copy_source_j = source_j
                                if type(source_j) == int:
                                    copy_source_j += v2
                                else:
                                    source_j = slice(v2, nmbj+v2, 2)

                                copy_source_k = source_k
                                if type(source_k) == int:
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

        # we should only have trouble populating boundaries when we extend beyond the domain. if
        # instead we end up printing a message here, there's something odd afoot.
        tvi = ti + di
        tvj = tj + dj
        tvk = tk + dk
        if tvi >= 0 and tvj >= 0 and tvk >= 0:
            pass

        return new_meshblock

    """
    @jit
    def another_metric(self, x):
        eta = jnp.asarray([[1,0,0],[0,1,0],[0,0,1]])
        a = self.bhspin
        aa = a * a
        zz = x[3]*x[3]
        kk = 0.5 * (x[1]*x[1] + x[2]*x[2] + zz - aa)
        rr = jnp.sqrt(kk * kk + aa * zz ) + kk
        r = jnp.sqrt(rr)
        f = (2.0 * rr * r)/(rr * rr + aa * zz)
        l = jnp.array([(r * x[1] + a * x[2])/(rr + aa) , (r* x[2] - a * x[1])/(rr + aa) , x[3]/r])
        return eta + f * (l[:,jnp.newaxis] * l[jnp.newaxis,:])

    @jit
    def vec_another_metric(self, X):
        return vmap(self.another_metric)(X)
        """

    """
    @jit
    def metric(x):
        '''
        !@brief Calculates the Kerr-schild metric in Cartesian Co-ordinates.
        @param x 1-D jax array with 4 elements corresponding to {t,x,y,z} which is equivalent to 4-position vector
        @returns a 4 X 4 two dimensional array reprenting the metric

        '''
        eta = jnp.asarray([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        a = a_spin
        aa = a * a
        zz = x[3]*x[3]
        kk = 0.5 * (x[1]*x[1] + x[2]*x[2] + zz - aa)
        rr = jnp.sqrt(kk * kk + aa * zz ) + kk
        r = jnp.sqrt(rr)
        f = (2.0 * rr * r)/(rr * rr + aa * zz)
        l = jnp.array([1, (r * x[1] + a * x[2])/(rr + aa) , (r* x[2] - a * x[1])/(rr + aa) , x[3]/r])
        return eta + f * (l[:,jnp.newaxis] * l[jnp.newaxis,:])

    @jit
    def imetric(self, x):
        '''
        !@brief Used by rhs()
        @param x The 4 vector position
        @returns The inverse of the metric at position x.

        '''
        return inv(self.metric(x))

    @jit
    def vec_metric(self, X):
        return vmap(self.metric)(X)

    @jit
    def vec_imetric(self, X):
        return vmap(self.imetric)(X)
        """

    def get_prims_from_geodesics(self, S, interp_method='linear', fluid_gamma=13./9):

        fluid_params = dict(fluid_gamma=fluid_gamma)

        nsteps, npx, _ = S.shape

        densff_data     = np.zeros((nsteps, npx))
        internal_u_data = np.zeros((nsteps, npx))
        U_1_data        = np.zeros((nsteps, npx))
        U_2_data        = np.zeros((nsteps, npx))
        U_3_data        = np.zeros((nsteps, npx))
        B_1_data        = np.zeros((nsteps, npx))
        B_2_data        = np.zeros((nsteps, npx))
        B_3_data        = np.zeros((nsteps, npx))

        populated = np.zeros((nsteps, npx))
        prims = np.zeros((nsteps, npx, 8))

        for mbi in tqdm(self.mb_index_map.values()):

            mb_x1min = self.x1f[mbi].min()
            mb_x1max = self.x1f[mbi].max()
            mb_x2min = self.x2f[mbi].min()
            mb_x2max = self.x2f[mbi].max()
            mb_x3min = self.x3f[mbi].min()
            mb_x3max = self.x3f[mbi].max()

            mb_mask = (mb_x1min < S[:,:,1]) & (S[:,:,1] <= mb_x1max)
            mb_mask &= (mb_x2min < S[:,:,2]) & (S[:,:,2] <= mb_x2max)
            mb_mask &= (mb_x3min < S[:,:,3]) & (S[:,:,3] <= mb_x3max)
            mb_mask &= (populated == 0)

            x1e = self.get_extended(self.x1v[mbi])
            x2e = self.get_extended(self.x2v[mbi])
            x3e = self.get_extended(self.x3v[mbi])

            # get meshblock key information
            tlevel = self.Levels[mbi]
            ti, tj, tk = self.LogicalLocations[mbi]
            key = tlevel, ti, tj, tk

            if np.count_nonzero(mb_mask) == 0:
                continue


            # create and use the interpolation object
            for nprm in range(self.nprim_all):
                prm = self.all_meshblocks[mbi,nprm, :, :, :]
                rgi = RegularGridInterpolator((x1e, x2e, x3e), prm.transpose((2,1,0)),
                                                method=interp_method)

                remapped = rgi((S[:,:,1][mb_mask], S[:,:,2][mb_mask], S[:,:,3][mb_mask]))
                outidx, outval = self.map_prim_to_prim(remapped, nprm, self.variable_names, fluid_params)

                prims[mb_mask, outidx] = outval

            # ensure we don't accidentally overwrite already-populated
            # cells (precision issues?)
            populated[mb_mask] = 1


        densff_data     = prims[..., 0]
        internal_u_data = prims[..., 1]
        U_1_data        = prims[..., 2]
        U_2_data        = prims[..., 3]
        U_3_data        = prims[..., 4]
        B_1_data        = prims[..., 5]
        B_2_data        = prims[..., 6]
        B_3_data        = prims[..., 7]

        del prims
        del populated

        primitive_data = dict(
            dens=densff_data,
            u=internal_u_data,
            U1=U_1_data,
            U2=U_2_data,
            U3=U_3_data,
            B1=B_1_data,
            B2=B_2_data,
            B3=B_3_data
        )

        return primitive_data

    def compute_tensorial(self, S, primitive_data):

        nsteps, npx, _ = S.shape

        UuUu = np.zeros((nsteps, npx))

        total_u = np.array([primitive_data['U1'], primitive_data['U2'], primitive_data['U3']])
        total_u = np.transpose(total_u, (1,2,0))

        for i in tqdm(range(nsteps)):
            UuUu[i,:] = ((vec_metric(S[i,:,:4], self.bhspin)[:,1:,1:] @ total_u[i,:,:].reshape(npx,3,1)).reshape(npx,1,3) @ total_u[i,:,:].reshape(npx,3,1)).reshape(npx)
        del total_u

        GAMMA = np.sqrt(1 + UuUu)
        del UuUu

        final_M = np.zeros((nsteps, npx, 4))
        for i in tqdm(range(0, nsteps)):
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
