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


def test_kerr_cks():

    saved_shadows = np.load("data/shadow_data.npy", allow_pickle=True).item()

    for key in saved_shadows.keys():

        bhspin = saved_shadows[key]['bhspin']
        inc = saved_shadows[key]['inclination']
        trial_angles = saved_shadows[key]['angles']
        target_radii = saved_shadows[key]['radii']

        radii = ma.find_shadow_bisection_angles(bhspin, inc, trial_angles)

        assert np.allclose(radii, target_radii, rtol=1e-2)


if __name__ == "__main__":

    test_kerr_cks()
