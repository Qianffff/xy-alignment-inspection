import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2
from skimage.measure import block_reduce
from Kernel_and_convolution import *
from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Function_graveyard import *
from Variables_and_constants import *

"""
The chip and all its features (like edges and alignment marks) are modeled using a grid. The grid 
has a size of pixels_x by pixels_y. Each pixel has a certain value, which is related to how many 
secondary electrons are released when a primary electron hits the wafer at the corresponding location.

For each pixel, a convolution is performed and the resulting number is related to how many secondary 
electrons are released when the Gaussian beam hits the pixel. Note that the beam, due to its profile, 
also has hits the surrounding pixels. The width of the Gaussian distribution is directly related to 
the probe size of the electron beam. The probe size is typically expressed in terms of its full width 
half maximum (FWHM), which can be obtained from the beam spot model by considering contributions from 
diffraction, spherical aberration, chromatic aberration, and source size. 

Once this convolution grid is calculated, we multiply the convoluted grid by the scanning time per 
pixel and the beam current, since the expected number of secondary electrons is proportional to these 
factors. We then have the expected number of secondary electrons for each pixel. We model shot noise 
using a Poisson distribution where the expectation value for each pixel is the expected number of 
secondary electrons emitted by that pixel.
"""


if __name__ == "__main__":


# ===================== Process image =====================

    # Histogram of errors in detected positions
    displacements = []

    for i in range(simulation_runs):  
        print(i)          
        # Generate wafer image
        grid, pixel_width_x, pixel_width_y, pixels_x, pixels_y, shift_x, shift_y, rotation = real_image(pixel_width_real_x,pixel_width_real_y,frame_width_x,frame_width_y,cross_length,cross_line_width)  
        print(f"Cross middle x pixel = {int(np.round(pixels_x/2+shift_x))}")
        print(f"Cross middle y pixel = {int(np.round(pixels_y/2+shift_y))}")

        # Resize image to go from real/original pixel width to measure pixel width
        grid = resample_image_by_pixel_size(grid, pixel_width_real_x, pixel_width_x)

        # Use Gaussian distribution to meassure image
        picture_grid, half_pixel_width_gaussian_kernel, sigma = measured_image(grid, pixel_width_x, pixel_width_y, beam_current, scan_time_per_pixel)

        # Denoise the image
        picture_grid_denoised = denoise_image(picture_grid)

        # Position of the cross
        centerx, centery, cross_points =cross_position(picture_grid_denoised,intensity_threshold)

        # Listing some values of variables used in the simulation
        time_to_make_picture = pixels_x*pixels_y*scan_time_per_pixel   
        # Compute displacement (Euclidean distance in pixels)
        dx = (centerx - int(np.round((pixels_x+1)/2+shift_x))) * pixel_width_x * 1e9  # nm
        dy = (centery - int(np.round((pixels_y+1)/2+shift_y))) * pixel_width_y * 1e9  # nm
        displacement = np.sqrt(dx**2 + dy**2)

        displacements.append(displacement)
    if simulation_runs >0:
        displacements = np.array(displacements)
        #plotting of histogram
        plt.figure(figsize=(8,6))
        plt.hist(displacements, bins=20, color='skyblue', edgecolor='k')
        plt.xlabel("Center shift (nm)")
        plt.ylabel("Counts")
        plt.title(f"Distribution of cross center shift\nBeam current = {beam_current*1e12:.1f} pA, runs = {simulation_runs}, time to make pictures = {time_to_make_picture}")
        plt.show()  

        # Write each item on a new line in the file
        #with open('output_Olivier.txt', 'w') as file:
        #    for item in displacements:
        #        file.write(str(item) + '\n')
   
        print('Mean error = ', np.mean(displacements))
        print('Standard deviation of error = ', np.std(displacements))




    # Generate wafer image
    grid, pixel_width_real_x, pixel_width_real_y, pixels_real_x, pixels_real_y, shift_real_x, shift_real_y, rotation = real_image(pixel_width_real_x,pixel_width_real_y,frame_width_x,frame_width_y,cross_length,cross_line_width)  
    print(f"Cross middle x real pixel = {int(np.round(pixels_real_x/2+shift_real_x))}")
    print(f"Cross middle y real pixel = {int(np.round(pixels_real_y/2+shift_real_y))}")
    print(f"Rotation = {rotation:.3f}")

    # Resize image to go from real/original pixel width to measure pixel width
    grid_resampled = resample_image_by_pixel_size(grid, pixel_width_real_x, pixel_width_x)

    # Use Gaussian distribution to meassure image
    picture_grid, half_pixel_width_gaussian_kernel, sigma = measured_image(grid_resampled, pixel_width_x, pixel_width_y, beam_current, scan_time_per_pixel)

    # Denoise the image
    picture_grid_denoised = denoise_image(picture_grid)

    # Position of the cross
    centerx, centery, cross_points = cross_position(picture_grid_denoised,intensity_threshold)
    # Difference between calculated cross center position and actual position (in m)
    absolute_distance_error = np.linalg.norm([int(np.round(pixels_real_x/2+shift_real_x)) - centerx*resize_factor,int(np.round(pixels_real_y/2+shift_real_y)) - centery*resize_factor])*pixel_width_real_x

    # Listing some values of variables used in the simulation
    time_to_make_picture = (pixels_real_x/resize_factor)*(pixels_real_y/resize_factor)*scan_time_per_pixel
    print(f"Time to make image = {time_to_make_picture:.5f} seconds")
    print(f"Scan time per pixel = {scan_time_per_pixel*1e6:.5f} µs")
    print(f"Absolute distance error = {absolute_distance_error*1e9:.3f} nm")
    print(f"Beam current = {beam_current*1e12} pA")
    print(f"Error std = {error_std*1e9:.3f} nm")

    # Angle of the cross
    black_white_grid = detect_and_plot_harris_corners(picture_grid_denoised,dot_radius=1,dot_alpha=0.25,k=0.24,percentile=intensity_threshold)
    if rotation_find_boolean == True:
        found_rotation = find_rotation(black_white_grid,shift_real_x,shift_real_y,cross_length=cross_length,cross_width=cross_line_width,frame_width_x=frame_width_x,frame_width_y=frame_width_y)
        print(f"Found rotation = {found_rotation}")
        print(f"Angle error = {found_rotation-rotation:.2f}")
    
# ===================== Plot =====================
    if show_plots == True:
        #Plot the grid of SE yields. This represents what the real wafer pattern looks like.
        plt.figure(figsize=(12,12))
        plt.imshow(grid)
        plt.title('Secondary electron yield grid')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.5)

        #Plot the Gaussian kernel
        #plot_kernel(half_pixel_width_gaussian_kernel,sigma)

        # Plotting of the meassured SEM image
        plt.figure(figsize=(12,12))
        plt.imshow(picture_grid)
        plt.title('Simulated SEM image')
        plt.colorbar()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        
        # Plotting the denoised
        plt.figure(figsize=(12,12))
        plt.imshow(picture_grid_denoised)
        plt.title('Simulated SEM image denoised')
        plt.colorbar()
        plt.scatter(centerx, centery, c='red', marker='+', s=200, label='Center')
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)
        plt.pause(0.5)

        # Plotting the I vs t
        #plt.figure()
        #plt.plot(beam_current_array,scan_time_per_image_array,"k.-")
        #plt.xlabel("Beam current (pA)")
        #plt.ylabel("Time per 1 µm² image (s)")
        #plt.show(block=True)
