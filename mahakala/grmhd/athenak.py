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

            if mbi < nstart or mbi >= nend:
                break

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

