from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Variables_and_constants import *

def simulate(procedure):
    grid, cross_center = real_image() # Generate the wafer area
    
    # Plot the grid of SE yields. This represents what the real wafer pattern looks like.
    plt.figure(figsize=(6,6))
    plt.imshow(grid)
    plt.title('Secondary electron yield grid')
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.5)
    
    steps = len(procedure)-1 # Number of steps in the procedure (counting from 0)
    step = 0 # Current step
    while step <= steps:
        frame_width, pixel_width, SNR = procedure[step] # Get the frame_width, pixel_width and SNR for the current step
        # Narrow the scanning area (except when it's the first step)
        if step != 0:
            frame_halfwidth = frame_width/pixel_width_real/2 # in pixels
            zoomed_grid = grid[int(np.round(max(0,cross_center_measured_px[0]-frame_halfwidth))):
                          int(np.round(min(pixels_real,cross_center_measured_px[0]+frame_halfwidth)))+1,
                          int(np.round(max(0,cross_center_measured_px[1]-frame_halfwidth))):
                          int(np.round(min(pixels_real,cross_center_measured_px[1]+frame_halfwidth)))+1]
        else: zoomed_grid = grid
        
        # Simulate the scanning of the image with the e-beam
        measured_image = measure_image(zoomed_grid,pixel_width,SNR)
        
        # Calculate the position of the center of the cross
        cross_center_measured_px = cross_position(measured_image,intensity_threshold)
        
        if step != 0:
            # Position of the origin of the zoomed grid relative to the previous grid (in pixels)
            zoomed_grid_origin = np.array([int(np.round(max(0,cross_center_measured_px[0]-frame_halfwidth))),int(np.round(max(0,cross_center_measured_px[1]-frame_halfwidth)))])
            # Translate cross position from image coordinates to wafer coordinates
            cross_center_measured_px += zoomed_grid_origin * (pixel_width/pixel_width_real)
        
        cross_center_measured = cross_center_measured_px * pixel_width # Convert from pixels to meters
        # Difference between calculated cross center position and actual position (in m)
        error = np.linalg.norm([cross_center[0] - cross_center_measured[0], cross_center[1] - cross_center_measured[1]])
        print(f"Error = {error*1e9:.3f} nm")
        
        # ===================== Plot =====================
        if show_plots == True:
            # Plotting the denoised
            plt.figure(figsize=(6,6))
            plt.imshow(denoise_image(measured_image))
            plt.title('Simulated SEM image denoised')
            plt.colorbar()
            plt.scatter(cross_center_measured_px[1], cross_center_measured_px[0], c='red', marker='+', s=200, label='Center')
            plt.legend()
            plt.tight_layout()
            plt.show(block = step==steps)
            plt.pause(0.5)
            
        step += 1 # Move to the next step of the procedure

# Simulate the alignment procedure
simulate(procedure)