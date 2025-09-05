import numpy as np

import numpy as np
from scipy.optimize import fsolve

def gaussian_beam(I_opt, sigma, size=256):
    """
    Generate a 2D Gaussian beam distribution.
    I_opt: optimal current
    sigma: standard deviation (determines the beam size)
    size: grid size (pixels)
    """
    x = np.linspace(-3*sigma, 3*sigma, size)
    y = np.linspace(-3*sigma, 3*sigma, size)
    X, Y = np.meshgrid(x, y)
    beam = I_opt * np.exp(-(X**2 + Y**2) / (2*sigma**2))
    beam /= beam.sum()  # normalize
    return beam
