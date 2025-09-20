import numpy as np
from Variables_and_constants import *

# Define the function which generates the Gauss kernel that is used in the convolution
def shifted_gauss1d(kernel_width,sigma,shift):
    x = np.arange(kernel_width)
    center = (kernel_width-1)/2 + shift  # shifted center
    return np.exp(-0.5*((x-center)/sigma)**2)

# Create the kernel
def gauss_kernel(kernel_width,sigma,shift):
    gauss1d_x = shifted_gauss1d(kernel_width,sigma,shift[0])
    gauss1d_y = shifted_gauss1d(kernel_width,sigma,shift[1])
    kernel = np.outer(gauss1d_x, gauss1d_y)
    if kernel.sum() != 0: kernel /= kernel.sum() # Normalize
    return kernel

def convolve_at_pixel(grid, kernel,pixel):
    kernel_width = kernel.shape[0] # kernel width (in pixels)
    kernel_halfwidth = int(np.round((kernel_width-1)/2))
    grid_width = grid.shape[0] # grid width (in pixels)
    
    # Compute the bounds of the patch in the grid, taking into account the left, right, top, and bottom boundary
    i,j = pixel
    patch_start_x = max(i - kernel_halfwidth, 0)
    patch_end_x   = min(i + kernel_halfwidth, grid_width-1)
    patch_start_y = max(j - kernel_halfwidth, 0)
    patch_end_y   = min(j + kernel_halfwidth, grid_width-1)
    # Extract the patch from the grid
    patch = grid[int(np.round(patch_start_x)) : int(np.round(patch_end_x+1)),
                 int(np.round(patch_start_y)) : int(np.round(patch_end_y+1))]
    # Now crop the kernel to match the patch
    kernel_start_x =  patch_start_x - (i - kernel_halfwidth)
    kernel_end_x   = kernel_start_x + patch.shape[0] -1

    kernel_start_y = patch_start_y - (j - kernel_halfwidth)
    kernel_end_y   = kernel_start_y + patch.shape[1] -1
    kernel_cropped = kernel[int(np.round(kernel_start_x)):int(np.round(kernel_end_x+1)),
                            int(np.round(kernel_start_y)):int(np.round(kernel_end_y+1))]

    kernel_cropped = kernel_cropped/kernel_cropped.sum()
    # Elementwise multiply and sum
    return np.sum(patch * kernel_cropped)