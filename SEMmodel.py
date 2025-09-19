from Kernel_and_convolution import *
from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Function_graveyard import *
from Variables_and_constants import *

# Generate wafer image
grid, pixel_width_real, pixels_real_x, pixels_real_y, shift_real_x, shift_real_y, rotation = real_image(pixel_width_real,frame_width_x,frame_width_y,cross_length,cross_line_width)  

# Resize image to go from real/original pixel width to measure pixel width
grid_resampled = resample_image_by_pixel_size(grid, pixel_width_real, pixel_width)

# Use Gaussian distribution to meassure image
picture_grid, half_pixel_width_gaussian_kernel, sigma = measured_image(grid_resampled, pixel_width, beam_current, scan_time_per_pixel)

# Denoise the image
picture_grid_denoised = denoise_image(picture_grid)

# Position of the cross
centerx, centery, cross_points = cross_position(picture_grid_denoised,intensity_threshold)
# Difference between calculated cross center position and actual position (in m)
absolute_distance_error = np.linalg.norm([int(np.round(pixels_real_x/2+shift_real_x)) - centerx*resize_factor,int(np.round(pixels_real_y/2+shift_real_y)) - centery*resize_factor])*pixel_width_real

# Listing some values of variables used in the simulation
time_to_make_picture = (pixels_real_x/resize_factor)*(pixels_real_y/resize_factor)*scan_time_per_pixel
print(f"Time to make image = {time_to_make_picture:.5f} seconds")
print(f"Scan time per pixel = {scan_time_per_pixel*1e6:.5f} µs")
print(f"Absolute distance error = {absolute_distance_error*1e9:.3f} nm")

# ===================== Plot =====================
if show_plots == True:
    #Plot the grid of SE yields. This represents what the real wafer pattern looks like.
    plt.figure(figsize=(12,12))
    plt.imshow(grid)
    plt.title('Secondary electron yield grid')
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.5)

    # Plotting of the meassured SEM image
    plt.figure(figsize=(12,12))
    plt.imshow(picture_grid)
    plt.title('Simulated SEM image')
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    
    # Plotting the white and black image
    black_white_grid = detect_and_plot_harris_corners(picture_grid_denoised,dot_radius=1,dot_alpha=0.25,k=0.24,percentile=intensity_threshold)

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

