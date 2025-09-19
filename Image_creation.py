import numpy as np
from scipy.ndimage import rotate, shift
from skimage.measure import block_reduce
from Kernel_and_convolution import *
from Variables_and_constants import *


def real_image(pixel_width=2e-9,frame_width_x=1e-6,frame_width_y=1e-6,cross_length=100e-9,cross_line_width=14e-9,shift_x=None,shift_y=None,rotation=None,background_noise=True):
    # The number of pixels in the x and the y direction 
    
    center_pixel_x = int(np.round((pixels_x+1)/2))
    center_pixel_y = int(np.round((pixels_y+1)/2))
    # Grid of secondary electron (SE) yields
    grid = np.zeros([pixels_x,pixels_y])
    
    # Dimensions in pixels
    cross_pixel_length_arm = int(np.round(cross_length/2/pixel_width))
    cross_pixel_halfwidth = int(np.round(cross_line_width/2/pixel_width))

    # Define some useful parameters about the cross geometry
    cross_pixel_left_side = int(np.round(center_pixel_x - cross_pixel_length_arm)-1)
    cross_pixel_right_side = int(np.round(center_pixel_x + cross_pixel_length_arm))
    cross_pixel_top_side = int(np.round(center_pixel_y - cross_pixel_length_arm)-1)
    cross_pixel_bottom_side = int(np.round(center_pixel_y + cross_pixel_length_arm))
    cross_pixel_half_width_plus = int(np.round(center_pixel_x + cross_pixel_halfwidth))
    cross_pixel_half_width_minus = int(np.round(center_pixel_x - cross_pixel_halfwidth)-1)

    # Random position and rotation cross
    max_shift_x = int(np.round(center_pixel_x - 10/8*cross_pixel_length_arm))
    max_shift_y = int(np.round(center_pixel_y - 10/8*cross_pixel_length_arm))

    if shift_x == None:
        shift_x = int(np.random.randint(-max_shift_x,max_shift_x))
    if shift_y == None:
        shift_y = int(np.random.randint(-max_shift_y,max_shift_y))
    if rotation == None:
        rotation = np.random.uniform(0,90)

    # First: create the cross in the middle of the grid
    # Create the vertical line
    grid[cross_pixel_top_side:cross_pixel_bottom_side,
         cross_pixel_half_width_minus:cross_pixel_half_width_plus] += SE_yield_cross
    grid[cross_pixel_top_side+1:cross_pixel_bottom_side-1,
         cross_pixel_half_width_minus+1:cross_pixel_half_width_plus-1] -= SE_yield_cross/2
    # Create the horizontal line
    grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,
         cross_pixel_left_side:cross_pixel_right_side] += SE_yield_cross
    grid[cross_pixel_half_width_minus+1:cross_pixel_half_width_plus-1,
         cross_pixel_left_side+1:cross_pixel_right_side-1] -= SE_yield_cross/2
    # Remove doubly counted region
    grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,
         cross_pixel_half_width_minus:cross_pixel_half_width_plus] -= SE_yield_cross
    grid[cross_pixel_half_width_minus+1:cross_pixel_half_width_plus-1,
         cross_pixel_half_width_minus+1:cross_pixel_half_width_plus-1] += SE_yield_cross/2



    # Second: rotate the cross
    grid = rotate(grid, angle=rotation, reshape=False, order=3, mode='constant', cval=0)
    # The rotate function may return numbers smaller than zero. This will become problematic
    # later on when using the Poisson distribution. Therefore we now set all negative numbers
    # (which are very small like -6e-144) to zero. This is justified because they are already
    # very small and since we can not have a negative number of secondary electrons
    grid[grid < 0] = 0

    # Third: shift the rotated cross
    grid = shift(grid, shift=(shift_y, shift_x), order=3, mode='constant', cval=0)

    # Fourth: add noise in background
    # Use the first to have some randomness, and the second for a constant yield
    if background_noise == True:
        grid += np.random.random([pixels_x,pixels_y])*background_noise_level

    return grid, pixel_width, pixels_x, pixels_y, shift_x, shift_y, rotation


def resample_image_by_pixel_size(img, original_pixel_size_nm, new_pixel_size_nm):
    factor = int(np.round(new_pixel_size_nm/original_pixel_size_nm))
    downsampled_img = block_reduce(img,block_size=(factor,factor),func=np.mean)
    return downsampled_img

def measured_image(real_image,pixel_width,beam_current=500e-12,scan_time_per_pixel=4e-7,error_std=8e-9):
    
    
    # Calculate the expected number of SE per pixel
    pixels_x = np.shape(real_image)[0]
    pixels_y = np.shape(real_image)[0]
    picture_grid = np.zeros((pixels_x, pixels_y))
    expected_number_of_secondary_electrons = np.zeros((pixels_x, pixels_y))

    
    #Defines the direction in which the beam drift occurs
    random_angle = np.random.uniform(0,2*np.pi)
    
    for i in range(pixels_x):

        for j in range(pixels_y):
            # Random beam landing position error in x and y direction (in pixels)
            error_shift_x = (np.random.normal(scale=error_std) + np.cos(random_angle)*np.abs((np.random.normal(scale=(i*pixels_x + j + (FOV_count-1) * pixels_x * pixels_y) * (drift_rate * scan_time_per_pixel))))) / pixel_width 
            error_shift_y = (np.random.normal(scale=error_std) + np.sin(random_angle)*np.abs((np.random.normal(scale=(i*pixels_x + j + (FOV_count-1) * pixels_x * pixels_y) * (drift_rate * scan_time_per_pixel))))) / pixel_width 
            # Create kernel
            kernel_ij, i_shift, j_shift = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
                                     sigma,error_shift_x,error_shift_y)
            i_convolve, j_convolve = int(np.round(i+i_shift)), int(np.round(j+j_shift))
            if i+i_shift < 0: i_convolve = 0
            if i+i_shift > pixels_x-1: i_convolve = pixels_x - 1
            if j+j_shift < 0: j_convolve = 0
            if j+j_shift > pixels_y-1: j_convolve = pixels_y - 1

            # Perform the convolution
            expected_number_of_secondary_electrons[i, j] = convolve_at_pixel(
                real_image, kernel_ij, i_convolve, j_convolve)
        # Progress bar
        if int(np.round((i-1)/pixels_x*100)) % 5 != 0:
            if int(np.round(i/pixels_x*100)) % 5 == 0:
                print(str(int(np.round(i/pixels_x*100)))+str("%"),end=" ")


    print("\n")
    expected_number_of_secondary_electrons *= beam_current/e * scan_time_per_pixel * escape_factor * collector_efficiency
    # If there is no background noise, some numbers may become smaller than 0.
    # This gives an error in the upcoming Poisson function
    expected_number_of_secondary_electrons[expected_number_of_secondary_electrons<0] = 0


    # Simulate detected electrons using Poisson statistics
    picture_grid = np.random.poisson(expected_number_of_secondary_electrons)
    return picture_grid, half_pixel_width_gaussian_kernel, sigma