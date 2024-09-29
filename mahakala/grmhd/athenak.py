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

from tqdm import tqdm


def get_extended(arr):
    arr2 = np.zeros(arr.size + 2)
    arr2[1:-1] = arr
    dx = arr[1] - arr[0]
    arr2[0] = arr2[1] - dx
    arr2[-1] = arr2[-2] + dx
    return arr2


def get_all_vals_unique(xs):
    all_xs = []
    for x in xs:
        all_xs += list(x)
    return np.array(sorted(list(set(all_xs))))


def load_athenak_meshblocks(grmhd_filename):

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
        variable_names = np.array(hfp.attrs['VariableNames'])
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


        all_x1s = get_all_vals_unique(x1v)
        all_x2s = get_all_vals_unique(x2v)
        all_x3s = get_all_vals_unique(x3v)

        extrema = np.abs(np.array([all_x1s.min(), all_x1s.max(), all_x2s.min(),
                                all_x2s.max(), all_x3s.min(), all_x3s.max()]))

        all_meshblocks = []
        for mbi in tqdm(mb_index_map.values()):

            # get edges for grid interpolator
            x1e = get_extended(x1v[mbi])
            x2e = get_extended(x2v[mbi])
            x3e = get_extended(x3v[mbi])

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
                        new_meshblock = athenak.get_new_meshblock_boundary(*mb_info, mb_index_map,
                                                                        new_meshblock, uov, B)

            all_meshblocks.append(new_meshblock)


        all_meshblocks = np.array(all_meshblocks)



def get_key_for_level(c_level, n_level, ti, tj, tk, di, dj, dk):
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


def get_slice_source(ntot, oddity):
    if oddity == 1:
        return slice(ntot//2, ntot)
    else:
        return slice(0, ntot//2)


def get_01_source(v, ntot, oddity):
    if oddity == 1:
        if v == 0:
            return ntot//2
        return -1
    else:
        if v == 0:
            return 0
        return ntot // 2 - 1


def get_new_meshblock_boundary(tlevel, ti, tj, tk, di, dj, dk, nprim, nmbi, nmbj, nmbk, mb_index_map, new_meshblock, uov, B):

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
    trial_key = get_key_for_level(tlevel, tlevel-1, ti, tj, tk, di, dj, dk)
    if trial_key in mb_index_map:

        nmb = mb_index_map[trial_key]

        _, newi, newj, newk = trial_key

        oddi = (ti+di) % 2
        oddj = (tj+dj) % 2
        oddk = (tk+dk) % 2

        source_i = get_01_source(0, nmbi, oddi) if di == 1 else (get_01_source(-1, nmbi, oddi) if di == -1 else get_slice_source(nmbi, oddi))
        source_j = get_01_source(0, nmbj, oddj) if dj == 1 else (get_01_source(-1, nmbj, oddj) if dj == -1 else get_slice_source(nmbj, oddj))
        source_k = get_01_source(0, nmbk, oddk) if dk == 1 else (get_01_source(-1, nmbk, oddk) if dk == -1 else get_slice_source(nmbk, oddk))

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
    trial_key = get_key_for_level(tlevel, tlevel+1, ti, tj, tk, di, dj, dk)
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
