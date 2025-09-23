import numpy as np
from scipy.ndimage import rotate, shift
from skimage.measure import block_reduce
from Kernel_and_convolution import *
from Variables_and_constants import *


def real_image(frame_width_px = pixels_real):
    
    grid = np.zeros([frame_width_px,frame_width_px]) # Initialize grid of SE yields
    center_pixel = int(np.round((frame_width_px+1)/2 -1)) # Define the center pixel of the grid (in both x and y) (in pixels)
    
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
    grid += np.random.random([frame_width_px,frame_width_px])*(2*average_SE_yield_background)
    
    # Define the position of the center of the cross relative to the top-left corner of the image (in m)
    cross_center_x = int(np.round(center_pixel+cross_shift[0]))*pixel_width_real
    cross_center_y = int(np.round(center_pixel+cross_shift[1]))*pixel_width_real
    cross_center = np.array([cross_center_x,cross_center_y])

    return grid, cross_center

# Resample the grid from the real pixel size to the image pixel size
def resample_image_by_pixel_size(grid, pixel_width):
    factor = int(np.round(pixel_width/pixel_width_real))
    downsampled_grid = block_reduce(grid,block_size=(factor,factor),func=np.mean)
    downsampled_grid = downsampled_grid[:-1,:-1] # Remove edge effects due to the downsampling
    return downsampled_grid

def measure_image(grid,pixel_width,SNR):
    
    # Resample to go from the real pixel size to the measured image pixel size
    image = resample_image_by_pixel_size(grid, pixel_width)
    
    # Initiate grid of expected number of SEs
    pixels = np.shape(image)[0]
    expected_number_of_secondary_electrons = np.zeros((pixels, pixels))

    # Define the direction in which beam drift occurs
    beam_drift_angle = np.random.uniform(0,2*np.pi)
    # Calculate scan time per pixel (needed for calculating how much beam drift has occured)
    scan_time_per_pixel = SNR**2/(SE_yield*escape_factor*collector_efficiency * (beam_current/e))
    
    for i in range(pixels):
        for j in range(pixels):
            
            # Create a random beam position error (in m). This error is made up of two parts:
            # the beam placement error with std = beam_placement_error_std, and the beam drift.
            beam_placement_error_angle = np.random.uniform(0,2*np.pi)
            beam_position_error = np.random.normal(scale=beam_placement_error_std)*np.array([np.cos(beam_placement_error_angle),np.sin(beam_placement_error_angle)])
            beam_drift = np.abs((np.random.normal(scale=((FOV_count-1)*pixels**2 + i*pixels + j) * (scan_time_per_pixel * drift_rate))))
            beam_position_error += beam_drift*np.array([np.cos(beam_drift_angle),np.sin(beam_drift_angle)])
            
            beam_position_error /= pixel_width # Convert from meters to pixels
            # Split the beam position error into two parts: the number of full pixels and the rest
            error_pixels = np.trunc(beam_position_error)
            error_rest = beam_position_error - error_pixels
            
            # Create kernel
            kernel_ij = gauss_kernel(kernel_width,sigma,error_rest)
            # Define the pixel that is at the kernel is at the center of the convolution
            convolve_pixel = np.array([i,j]) + error_rest
            if i + error_pixels[0] < 0: convolve_pixel[0] = 0
            if i + error_pixels[0] > pixels-1: convolve_pixel[0] = pixels - 1
            if j + error_pixels[1] < 0: convolve_pixel[1] = 0
            if j + error_pixels[1] > pixels-1: convolve_pixel[1] = pixels - 1

            # Perform the convolution
            expected_number_of_secondary_electrons[i,j] = convolve_at_pixel(image,kernel_ij,convolve_pixel)
        
        # Progress bar
        if int(np.round((i-1)/pixels*100)) % 5 != 0:
            if int(np.round(i/pixels*100)) % 5 == 0:
                print(str(int(np.round(i/pixels*100)))+str("%"),end=" ")
    print()
    expected_number_of_secondary_electrons *= (beam_current/e) * scan_time_per_pixel * escape_factor*collector_efficiency
    
    # If there is no background noise, some numbers may become smaller than 0.
    # This gives an error in the upcoming Poisson function
    expected_number_of_secondary_electrons[expected_number_of_secondary_electrons<0] = 0

    # Add shot noise
    measured_image = np.random.poisson(expected_number_of_secondary_electrons)
    return measured_image