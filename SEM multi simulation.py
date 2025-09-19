from Kernel_and_convolution import *
from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Function_graveyard import *
from Variables_and_constants import *

def simulation(SNR,pixel_width,frame_width_x,frame_width_y,picture_block):
    scan_time_per_pixel = SNR**2/(SE_yield * escape_factor * collector_efficiency * (beam_current/e)) # Scan time per pixel (in s) (inverse of the scan rate)

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
    absolute_distance_error = np.linalg.norm([int(np.round(pixels_real_x/2+shift_real_x)) - centerx*resize_factor,int(np.round(pixels_real_y/2+shift_real_y)) - centery*resize_factor])*pixel_width_real_x

    # Listing some values of variables used in the simulation
    time_to_make_picture = (pixels_real_x/resize_factor)*(pixels_real_y/resize_factor)*scan_time_per_pixel
    print(f"Time to make image = {time_to_make_picture:.5f} seconds")
    print(f"Scan time per pixel = {scan_time_per_pixel*1e6:.5f} µs")
    print(f"Absolute distance error = {absolute_distance_error*1e9:.3f} nm")

    # ===================== Plot =====================
    if show_plots == True:
        # Plotting the denoised
        plt.figure(figsize=(12,12))
        plt.imshow(picture_grid_denoised)
        plt.title('Simulated SEM image denoised')
        plt.colorbar()
        plt.scatter(centerx, centery, c='red', marker='+', s=200, label='Center')
        plt.legend()
        plt.tight_layout()
        plt.show(block=picture_block)
        plt.pause(0.5)
    return time_to_make_picture

# First the low-detail picture
SNR = 5, pixel_width = 20e-9, frame_width_x = 8e-6, frame_width_y = 8e-6, picture_block = False
time_to_make_picture_low_detail = simulation(SNR,pixel_width,frame_width_x,frame_width_y,picture_block)

# Second the high-detail picture
SNR = 10, pixel_width = 5e-9, frame_width_x = 1e-6, frame_width_y = 1e-6, picture_block = True
time_to_make_picture_high_detail = simulation(SNR,pixel_width,frame_width_x,frame_width_y,picture_block)