from Kernel_and_convolution import *
from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Variables_and_constants import *

# Histogram of errors in detected positions
displacements = []

for i in range(simulation_runs):  
    print(i)          
    # Generate wafer image
    grid, cross_center = real_image(frame_width)  
    print(f"Cross middle x pixel = {int(np.round(cross_center[0]/pixel_width))}")
    print(f"Cross middle y pixel = {int(np.round(cross_center[1]/pixel_width))}")

    # Resize image to go from real/original pixel width to measure pixel width
    grid = resample_image_by_pixel_size(grid, pixel_width)

    # Use Gaussian distribution to meassure image
    picture_grid = measured_image(grid, pixel_width,SNR)

    # Denoise the image
    picture_grid_denoised = denoise_image(picture_grid)

    # Position of the cross
    cross_center_measured_px = cross_position(picture_grid_denoised,intensity_threshold)
    cross_center_measured = cross_center_measured_px * pixel_width # Convert from pixels to meters  
    # Compute displacement (in nm)
    displacement = np.linalg.norm([cross_center[0] - cross_center_measured[0], cross_center[1] - cross_center_measured[1]]) * 1e9
    
    # Listing some values of variables used in the simulation
    time_to_make_picture = pixels**2*scan_time_per_pixel 
    
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
    #with open('output.txt', 'w') as file:
    #    for item in displacements:
    #        file.write(str(item) + '\n')

    print('Mean error = ', np.mean(displacements))
    print('Standard deviation of error = ', np.std(displacements))