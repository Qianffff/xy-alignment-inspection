from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Variables_and_constants import *

def simulate(procedure):
    frame_width = procedure[0][0] # Get the framewidth of the first step of the procedure
    grid, cross_center = real_image(frame_width) # Generate the wafer area
    
    # Plot the grid of SE yields. This represents what the real wafer pattern looks like.
    plt.figure(figsize=(12,12))
    plt.imshow(grid)
    plt.title('Secondary electron yield grid')
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.5)
    
    steps = len(procedure)-1 # Number of steps in the procedure (counting from 0)
    step = 0 # Current step
    while step <= steps:
        
        frame_width = procedure[step][0] # Get the frame_width for the current step
        # Narrow the scanning area (except when it's the first step)
        if step != 0:
            zoomed_grid = grid[int(np.round(max(0,cross_center_measured_px[0]-frame_width/pixel_width_real/2))):
                          int(np.round(min(pixels_real,cross_center_measured_px[0]+frame_width/pixel_width_real/2))),
                          int(np.round(max(0,cross_center_measured_px[1]-frame_width/pixel_width_real/2))):
                          int(np.round(min(pixels_real,cross_center_measured_px[1]+frame_width/pixel_width_real/2)))]
        else: zoomed_grid = grid
        pixel_width, SNR = procedure[step][1], procedure[step][2] # Get the pixel_width and SNR for the current step
        # Resample grid to go from the real pixel width to the image pixel width
        grid_resampled = resample_image_by_pixel_size(zoomed_grid, pixel_width)
        # Simulate the scanning of the image with the e-beam
        picture_grid = measured_image(grid_resampled,pixel_width,SNR)
        # Denoise the image
        picture_grid_denoised = denoise_image(picture_grid)
        
        # Calculate the position of the center of the cross
        cross_center_measured_px = cross_position(picture_grid_denoised,intensity_threshold)
        cross_center_measured = cross_center_measured_px * pixel_width # Convert from pixels to meters
        # Difference between calculated cross center position and actual position (in m)
        # cross_center still needs to be converted from real grid coordinates to zoomed grid coordinates!!!
        absolute_distance_error = np.linalg.norm([cross_center[0] - cross_center_measured[0], cross_center[1] - cross_center_measured[1]])
        print(f"Absolute distance error = {absolute_distance_error*1e9:.3f} nm")
        
        # ===================== Plot =====================
        if show_plots == True:
            # Plotting the denoised
            plt.figure(figsize=(12,12))
            plt.imshow(picture_grid_denoised)
            plt.title('Simulated SEM image denoised')
            plt.colorbar()
            plt.scatter(cross_center_measured_px[0], cross_center_measured_px[1], c='red', marker='+', s=200, label='Center')
            plt.legend()
            plt.tight_layout()
            plt.show(block = step==steps)
            plt.pause(0.5)
            
        step += 1 # Move to the next step of the procedure

# Simulate the alignment procedure
simulate(procedure)