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
    grid, pixel_width, pixels_x, pixels_y, shift_x, shift_y, rotation = real_image(pixel_width_real,frame_width_x,frame_width_y,cross_length,cross_line_width)  
    print(f"Cross middle x pixel = {int(np.round(pixels_x/2+shift_x))}")
    print(f"Cross middle y pixel = {int(np.round(pixels_y/2+shift_y))}")

    # Resize image to go from real/original pixel width to measure pixel width
    grid = resample_image_by_pixel_size(grid, pixel_width_real, pixel_width)

    # Use Gaussian distribution to meassure image
    picture_grid, half_pixel_width_gaussian_kernel, sigma = measured_image(grid, pixel_width, beam_current, scan_time_per_pixel)

    # Denoise the image
    picture_grid_denoised = denoise_image(picture_grid)

    # Position of the cross
    centerx, centery, cross_points =cross_position(picture_grid_denoised,intensity_threshold)

    # Listing some values of variables used in the simulation
    time_to_make_picture = pixels_x*pixels_y*scan_time_per_pixel   
    # Compute displacement (Euclidean distance in pixels)
    dx = (centerx - int(np.round((pixels_x+1)/2+shift_x))) * pixel_width * 1e9  # nm
    dy = (centery - int(np.round((pixels_y+1)/2+shift_y))) * pixel_width * 1e9  # nm
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
    #with open('output.txt', 'w') as file:
    #    for item in displacements:
    #        file.write(str(item) + '\n')

    print('Mean error = ', np.mean(displacements))
    print('Standard deviation of error = ', np.std(displacements))