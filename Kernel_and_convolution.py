import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2
from skimage.measure import block_reduce
from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Function_graveyard import *
from Variables_and_constants import *

# Define the function which generates the Gauss kernel that is used in the convolution
def shifted_gauss1d(gauss_pixels,sigma,shift):
    x = np.arange(gauss_pixels)
    center = (gauss_pixels - 1) / 2 + shift  # shifted center
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def gauss_kernel(gauss_pixels,sigma,shift_x=0,shift_y=0):
    shift_x_rest = shift_x - np.trunc(shift_x)
    shift_y_rest = shift_y - np.trunc(shift_y)
    shift_x_pixel = np.trunc(shift_x)
    shift_y_pixel = np.trunc(shift_y)
    gauss1d_x = shifted_gauss1d(gauss_pixels,sigma,shift_x_rest)
    gauss1d_y = shifted_gauss1d(gauss_pixels,sigma,shift_y_rest)
    kernel = np.outer(gauss1d_x, gauss1d_y)
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    if kernel.sum() == 0:
        kernel = kernel
    return kernel, shift_x_pixel, shift_y_pixel

# The standard functions in scipy/numpy implemented the convolution function
# to perform the convolution on the entire array, while we want to do it on a 
# per pixel basis since we have a different kernel for each pixel (to simulate beam position error).
def convolve_at_pixel(grid, kernel, i, j):
    kh, kw = kernel.shape # kernel height and width
    ph, pw = kh // 2, kw // 2  # patch height and width
    
    # Grid dimensions
    h, w = grid.shape

    # Compute the bounds of the patch in the grid, taking into account the 
    # left, right, bottom and top boundary
    i_start = max(i - ph, 0)
    i_end   = min(i + ph, h)

    j_start = max(j - pw, 0)
    j_end   = min(j + pw, w)
    
    # Extract the patch from the grid
    patch = grid[i_start:i_end, j_start:j_end]

    # Now crop the kernel accordingly to match the patch
    k_i_start = ph - (i - i_start)
    k_i_end   = k_i_start + patch.shape[0] 

    k_j_start = pw - (j - j_start)
    k_j_end   = k_j_start + patch.shape[1]
    kernel_cropped = kernel[k_i_start:k_i_end, k_j_start:k_j_end]

    kernel_cropped = kernel_cropped/kernel_cropped.sum()
    # Elementwise multiply and sum
    return np.sum(patch * kernel_cropped)