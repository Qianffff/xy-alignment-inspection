import numpy as np
from scipy.ndimage import rotate, shift
from skimage.measure import block_reduce
from Kernel_and_convolution import *
from Variables_and_constants import *


def real_image():
    
    grid = np.zeros([pixels_real,pixels_real]) # Initialize grid of SE yields
    center_pixel = int(np.round((pixels_real+1)/2 -1)) # Define the center pixel of the grid (in both x and y) (in pixels)
    
    # Cross dimensions in pixels
    cross_length_px = int(cross_length/pixel_width_real)
    if cross_length_px % 2 == 0: cross_length_px += 1
    
    cross_linewidth_px = int(cross_linewidth/pixel_width_real)
    if cross_linewidth_px % 2 == 0: cross_linewidth_px += 1
    
    # Define some useful parameters about the cross geometry
    cross_arm_length_px = int(np.round((cross_length_px - 1)/2))
    cross_half_linewidth_px = int(np.round((cross_linewidth_px - 1)/2))
    
    cross_left_edge_px = int(np.round(center_pixel - cross_arm_length_px))
    cross_right_edge_px = int(np.round(center_pixel + cross_arm_length_px))
    cross_top_edge_px = int(np.round(center_pixel - cross_arm_length_px))
    cross_bottom_edge_px = int(np.round(center_pixel + cross_arm_length_px))
    cross_halfwidth_plus_px = int(np.round(center_pixel + cross_half_linewidth_px))
    cross_halfwidth_minus_px = int(np.round(center_pixel - cross_half_linewidth_px))

    # Generate a random cross position (in pixels) and rotation (in degrees)
    max_shift = int(np.round(center_pixel - cross_arm_length_px))
    cross_shift = [int(np.random.randint(-max_shift,max_shift)),int(np.random.randint(-max_shift,max_shift))]
    rotation = np.random.uniform(0,90)

    # Create the cross in the middle of the grid
    # Create the vertical line
    grid[cross_top_edge_px:cross_bottom_edge_px+1,
         cross_halfwidth_minus_px:cross_halfwidth_plus_px+1] += SE_yield_cross_edge
    grid[cross_top_edge_px + 1:cross_bottom_edge_px+1 - 1,
         cross_halfwidth_minus_px + 1:cross_halfwidth_plus_px+1 - 1] += -SE_yield_cross_edge + SE_yield_cross_body
    # Create the horizontal line
    grid[cross_halfwidth_minus_px:cross_halfwidth_plus_px+1,
         cross_left_edge_px:cross_right_edge_px+1] += SE_yield_cross_edge
    grid[cross_halfwidth_minus_px + 1:cross_halfwidth_plus_px+1 - 1,
         cross_left_edge_px + 1:cross_right_edge_px+1 - 1] += -SE_yield_cross_edge + SE_yield_cross_body
    # Fix doubly counted region
    grid[cross_halfwidth_minus_px:cross_halfwidth_plus_px+1,
         cross_halfwidth_minus_px:cross_halfwidth_plus_px+1] -= SE_yield_cross_edge
    grid[cross_halfwidth_minus_px + 1:cross_halfwidth_plus_px+1 - 1,
         cross_halfwidth_minus_px + 1:cross_halfwidth_plus_px+1 - 1] += SE_yield_cross_edge - SE_yield_cross_body

    # Second: rotate the cross
    grid = rotate(grid, angle=rotation, reshape=False, order=3, mode='constant', cval=0)
    # The rotate function may return numbers smaller than zero. This will become problematic
    # later on when using the Poisson distribution. Therefore we now set all negative numbers
    # (which are very small like -6e-144) to zero. This is justified because they are already
    # very small and since we can not have a negative number of secondary electrons
    grid[grid < 0] = 0

    # Third: shift the rotated cross
    grid = shift(grid, shift=cross_shift, order=3, mode='constant', cval=0)

    # Fourth: add noise in background
    grid += np.random.random([pixels_real,pixels_real])*(2*average_SE_yield_background)
    
    # Define the position of the center of the cross relative to the top-left corner of the image (in m)
    cross_center_x = int(np.round(center_pixel+cross_shift[0]))*pixel_width_real
    cross_center_y = int(np.round(center_pixel+cross_shift[1]))*pixel_width_real
    cross_center = [cross_center_x,cross_center_y]

    return grid, cross_center

# Resample the grid from the real pixel size to the image pixel size
def resample_image_by_pixel_size(grid, pixel_width):
    factor = int(np.round(pixel_width/pixel_width_real))
    downsampled_grid = block_reduce(grid,block_size=(factor,factor),func=np.mean)
    downsampled_grid = downsampled_grid[:-1,:-1] # Remove edge effects due to the downsampling
    return downsampled_grid

def measured_image(real_image,pixel_width,SNR):
    
    # Resample to go from the real pixel size to the measured image pixel size
    image = resample_image_by_pixel_size(real_image, pixel_width)
    
    scan_time_per_pixel = SNR**2/(SE_yield*escape_factor*collector_efficiency * (beam_current/e))
    
    # Initiate grid of expected number of SEs
    pixels = np.shape(image)[0]
    picture_grid = np.zeros((pixels, pixels))
    expected_number_of_secondary_electrons = np.zeros((pixels, pixels))

    # Define the direction in which beam drift occurs
    random_angle = np.random.uniform(0,2*np.pi)
    
    for i in range(pixels):
        for j in range(pixels):
            # Random beam landing position error in x and y direction (in pixels)
            error_shift_x = (np.random.normal(scale=error_std) + np.cos(random_angle)*np.abs((np.random.normal(scale=(i*pixels + j + (FOV_count-1) * pixels**2) * (drift_rate * scan_time_per_pixel))))) / pixel_width 
            error_shift_y = (np.random.normal(scale=error_std) + np.sin(random_angle)*np.abs((np.random.normal(scale=(i*pixels + j + (FOV_count-1) * pixels**2) * (drift_rate * scan_time_per_pixel))))) / pixel_width 
            # Create kernel
            kernel_ij, i_shift, j_shift = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
                                     sigma,error_shift_x,error_shift_y)
            i_convolve, j_convolve = int(np.round(i+i_shift)), int(np.round(j+j_shift))
            if i+i_shift < 0: i_convolve = 0
            if i+i_shift > pixels-1: i_convolve = pixels - 1
            if j+j_shift < 0: j_convolve = 0
            if j+j_shift > pixels-1: j_convolve = pixels - 1

            # Perform the convolution
            expected_number_of_secondary_electrons[i, j] = convolve_at_pixel(
                image, kernel_ij, i_convolve, j_convolve)
        # Progress bar
        if int(np.round((i-1)/pixels*100)) % 5 != 0:
            if int(np.round(i/pixels*100)) % 5 == 0:
                print(str(int(np.round(i/pixels*100)))+str("%"),end=" ")
                
    print("\n")
    
    expected_number_of_secondary_electrons *= (beam_current/e) * scan_time_per_pixel * escape_factor*collector_efficiency
    
    # If there is no background noise, some numbers may become smaller than 0.
    # This gives an error in the upcoming Poisson function
    expected_number_of_secondary_electrons[expected_number_of_secondary_electrons<0] = 0


    # Simulate detected electrons using Poisson statistics
    picture_grid = np.random.poisson(expected_number_of_secondary_electrons)
    return picture_grid