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

from mahakala.geodesics import find_shadow_bisection
from mahakala.geodesics import find_shadow_bisection_angles
from mahakala.geodesics import geodesic_integrator
from mahakala.geodesics import initialize_geodesics_at_camera

from mahakala.transfer import synchrotron_coefficients
from mahakala.transfer import solve_specific_intensity
from mahakala.transfer import solve_attenuated_emissivity

from jax.lib import xla_bridge
print('jax is using the', xla_bridge.get_backend().platform)

__all__ = [
    "find_shadow_bisection",
    "find_shadow_bisection_angles",
    "geodesic_integrator",
    "initialize_geodesics_at_camera",
    "synchrotron_coefficients",
    "solve_specific_intensity",
    "solve_attenuated_emissivity"
]
