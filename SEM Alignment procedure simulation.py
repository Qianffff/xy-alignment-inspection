from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Variables_and_constants import *

def simulate(procedure):
    # Generate the wafer area
    frame_width = procedure[0][0]
    frame_width_px = int(np.round((frame_width/pixel_width_real))) 
    if frame_width_px % 2 == 0 : frame_width_px += 1
    
    pixels_real = int(frame_width/pixel_width_real)
    if pixels_real % 2 == 0: pixels_real +=1
    
    grid, cross_center = real_image(frame_width_px)
    
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
        # Zoom in (except for the first step)
        if step == 0:
            zoomed_grid = grid # No zooming happens in the first step
            origin_shift = np.array([0,0]) # Origin doesn't shift in the first step
        else:
            # Zoom in around the cross position that was found in the previous step
            frame_halfwidth = frame_width/pixel_width_real/2 # in real pixels
            zoomed_grid = grid[int(np.round(max(0,cross_center_measured_rlpx[0]-frame_halfwidth))):
                          int(np.round(min(pixels_real,cross_center_measured_rlpx[0]+frame_halfwidth)))+1,
                          int(np.round(max(0,cross_center_measured_rlpx[1]-frame_halfwidth))):
                          int(np.round(min(pixels_real,cross_center_measured_rlpx[1]+frame_halfwidth)))+1]
            # Define the origin of current frame relative to the origin of the real grid (in real pixels)
            origin_shift = np.array([int(np.round(max(0,cross_center_measured_rlpx[0]-frame_halfwidth))),int(np.round(max(0,cross_center_measured_rlpx[1]-frame_halfwidth)))])
            
            # Slice the grid to make it square (It should still have the cross in it, but I'm not sure it currently always does)
            x,y = np.shape(zoomed_grid)
            if x > y:
                zoomed_grid = zoomed_grid[int((x-y)/2):int((x+y)/2),:]
                origin_shift += np.array([int((x-y)/2),0]) # Update the origin shift
            elif y > x:
                zoomed_grid = zoomed_grid[:,int((y-x)/2):int((x+y)/2)]
                origin_shift += np.array([0,int((y-x)/2)]) # Updat the origin shift
        
        # Simulate the scanning of the image with the e-beam
        measured_image = measure_image(zoomed_grid,pixel_width,SNR)
        
        # Calculate the position of the center of the cross (in image pixels)
        cross_center_measured_impx = cross_position(measured_image,intensity_threshold)
        cross_center_measured_rlpx = cross_center_measured_impx * (pixel_width/pixel_width_real) # Convert from image pixels to real pixels
        cross_center_measured_rlpx += origin_shift # Convert from image coordinates to real coordinates
        
        cross_center_measured = cross_center_measured_rlpx * pixel_width_real # Convert from pixels to meters
        # Difference between calculated cross center position and actual position (in m)
        error = np.linalg.norm([cross_center[0] - cross_center_measured[0], cross_center[1] - cross_center_measured[1]])
        print(f"Error = {error*1e9:.3f} nm")
        print()
        
        # ===================== Plot =====================
        if show_plots == True:
            # Plotting the denoised
            plt.figure(figsize=(6,6))
            plt.imshow(measured_image)
            plt.title('Simulated SEM image denoised')
            plt.colorbar()
            plt.scatter(cross_center_measured_impx[1], cross_center_measured_impx[0], c='red', marker='+', s=200, label='Measured center')
            plt.legend()
            plt.tight_layout()
            plt.show(block = step==steps)
            plt.pause(0.5)
            
        step += 1 # Move to the next step of the procedure

def show_time(procedure): 
    total_time = 0
    for i in range(len(procedure)):
        frame_width, pixel_width, SNR = procedure[i]

        scan_time_per_pixel = SNR**2/(SE_yield*escape_factor*collector_efficiency * (beam_current/e))

        pixels = frame_width / pixel_width

        time = scan_time_per_pixel*pixels**2 + beam_overhead_rate*(pixels*pixel_width)*(pixels-1)

        if i == 0: time = time * n_eFOVs * (ebeam_FOV_width**2/frame_width**2) + stage_overhead_time + latency # Account for taking multiple ebeam FOV sized images in the first step to find the mark
        else: time += latency
        total_time += time

    return total_time


simulate_bool = False
show_time_bool = True

if simulate_bool:
    # Simulate the alignment procedure
    simulate(procedure)
if show_time_bool:
    # Show the relevant numbers for each step
    print(f"Total alignment time = {show_time(procedure)*1e3:.6f} ms")