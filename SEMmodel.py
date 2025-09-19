import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2
from skimage.measure import block_reduce

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
def real_image(pixel_width_x=2e-9,pixel_width_y=2e-9,frame_width_x=1e-6,frame_width_y=1e-6,cross_length=100e-9,cross_line_width=14e-9,shift_x=None,shift_y=None,rotation=None,background_noise=True):
    # The number of pixels in the x and the y direction 
    pixels_x = int(np.rint(frame_width_x/pixel_width_x))
    pixels_y = int(np.rint(frame_width_y/pixel_width_y))

    if pixels_x % 2 == 0:
        pixels_x += 1
    
    if pixels_y % 2 == 0:
        pixels_y += 1
    
    center_pixel_x = int(np.round((pixels_x+1)/2))
    center_pixel_y = int(np.round((pixels_y+1)/2))
    # Grid of secondary electron (SE) yields
    grid = np.zeros([pixels_x,pixels_y])
    
    # Dimensions in pixels
    cross_pixel_length_arm = int(np.round(cross_length/2/pixel_width_x))
    cross_pixel_halfwidth = int(np.round(cross_line_width/2/pixel_width_x))

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

    return grid, pixel_width_x, pixel_width_y, pixels_x, pixels_y, shift_x, shift_y, rotation

def calculate_original_pixel_size(new_pixel_size_nm,original_pixel_size_nm_maximum):
    original_pixel_size_nm = new_pixel_size_nm
    while original_pixel_size_nm > original_pixel_size_nm_maximum:
        original_pixel_size_nm = original_pixel_size_nm/2
    return original_pixel_size_nm

def resample_image_by_pixel_size(img, original_pixel_size_nm, new_pixel_size_nm):
    factor = int(np.round(new_pixel_size_nm/original_pixel_size_nm))
    downsampled_img = block_reduce(img,block_size=(factor,factor),func=np.mean)
    return downsampled_img

def measured_image(real_image,pixel_width_x,pixel_width_y,beam_current=500e-12,scan_time_per_pixel=4e-7,error_std=8e-9):
    # Calculate the expected number of SE per pixel
    pixels_x = np.shape(real_image)[0]
    pixels_y = np.shape(real_image)[0]
    picture_grid = np.zeros((pixels_x, pixels_y))
    expected_number_of_secondary_electrons = np.zeros((pixels_x, pixels_y))
    
    # Calculate the beam width given the beam current
    FWHM = 8e-9 # (in m)
    sigma = FWHM/(2*np.sqrt(2*np.log(2))) # (in m)
    sigma = sigma/pixel_width_x # (in px)
    half_pixel_width_gaussian_kernel = int(np.ceil(3*sigma)) # (in px)
    
    
    
    for i in range(pixels_x):

        for j in range(pixels_y):
            # Random beam landing position error in x and y direction (in pixels)
            error_shift_x = np.random.normal(scale=error_std) / pixel_width_x
            error_shift_y = np.random.normal(scale=error_std) / pixel_width_y
            # Create kernel
            kernel_ij = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
                                     sigma,error_shift_x,error_shift_y)
            # Perform the convolution
            expected_number_of_secondary_electrons[i, j] = convolve_at_pixel(
                real_image, kernel_ij, i, j)
        # Progress bar
        if int(np.round(i/pixels_x*100000)) % 5000 == 0:
            print(str(int(np.round(i/pixels_x*100)))+str("%"),end=" ")

    
    expected_number_of_secondary_electrons *= beam_current/e * scan_time_per_pixel * escape_factor * collector_efficiency
    # If there is no background noise, some numbers may become smaller than 0.
    # This gives an error in the upcoming Poisson function
    expected_number_of_secondary_electrons[expected_number_of_secondary_electrons<0] = 0


    # Simulate detected electrons using Poisson statistics
    picture_grid = np.random.poisson(expected_number_of_secondary_electrons)
    return picture_grid, half_pixel_width_gaussian_kernel, sigma

# Define the function which generates the Gauss kernel that is used in the convolution
def shifted_gauss1d(gauss_pixels,sigma,shift):
    x = np.arange(gauss_pixels)
    center = (gauss_pixels - 1) / 2 + shift  # shifted center
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def gauss_kernel(gauss_pixels,sigma,shift_x=0,shift_y=0):
    gauss1d_x = shifted_gauss1d(gauss_pixels,sigma,shift_x)
    gauss1d_y = shifted_gauss1d(gauss_pixels,sigma,shift_y)
    kernel = np.outer(gauss1d_x, gauss1d_y)
    kernel = kernel / kernel.sum()
    return kernel

# The standard functions in scipy/numpy implemented the convolution function
# to perform the convolution on the entire array, while we want to do it on a 
# per pixel basis since we have a different kernel for each pixel (to simulate beam position error).
def convolve_at_pixel(grid, kernel, i, j):
    kh, kw = kernel.shape # kernel height and width
    ph, pw = kh // 2, kw // 2  # patch height and width
    
    # Grid dimensions
    h, w = grid.shape

    # Compute the bounds of the patch in the grid, taking into account the 
    # left, right, bottom and top boundary
    i_start = max(i - ph, 0)
    i_end   = min(i + ph, h)

    j_start = max(j - pw, 0)
    j_end   = min(j + pw, w)
    
    # Extract the patch from the grid
    patch = grid[i_start:i_end, j_start:j_end]

    # Now crop the kernel accordingly to match the patch
    k_i_start = ph - (i - i_start)
    k_i_end   = k_i_start + patch.shape[0] 

    k_j_start = pw - (j - j_start)
    k_j_end   = k_j_start + patch.shape[1]
    kernel_cropped = kernel[k_i_start:k_i_end, k_j_start:k_j_end]

    # Elementwise multiply and sum
    return np.sum(patch * kernel_cropped)

def plot_kernel(half_pixel_width_gaussian_kernel,sigma,shift_x=0,shift_y=0):
    # Plot the kernel
    plt.figure()
    plt.imshow(gauss_kernel(half_pixel_width_gaussian_kernel*2+1, sigma,shift_x,shift_y))
    plt.title('Beam profile')
    plt.xlabel('px')
    plt.ylabel('px')
    plt.show(block=False)
    plt.pause(0.5)
    return

# Transformations from Poisson noise to Gaussian noise and back
def anscombe_transform(image):
    return 2.0 * np.sqrt(image + 3.0 / 8.0)

def inverse_anscombe_transform(transformed):
    return (transformed / 2.0) ** 2 - 3.0 / 8.0

def denoise_image(image):
    transformed = anscombe_transform(image)
    sigma_est = estimate_sigma(transformed, channel_axis=None)

    denoised_transformed = denoise_nl_means(
        transformed,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None)
    denoised_image = inverse_anscombe_transform(denoised_transformed)
    return denoised_image



def calculate_CNR(picture_grid,
                  cross_pixel_left_side,cross_pixel_right_side,
                  cross_pixel_top_side,cross_pixel_bottom_side,
                  cross_pixel_half_width_plus,cross_pixel_half_width_minus):
    # Calculate contrast to noise ratio (CNR)

    cross_sum = 0 # Sum of secondary electron yields of all pixels in the cross
    cross_pixels = 0 # Number of pixels in the cross
    # Sum up the verticle line
    cross_sum += np.sum(picture_grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    cross_pixels += np.size(picture_grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    # Sum up the horizontal line
    cross_sum += np.sum(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side]) 
    cross_pixels += np.size(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side])
    # Remove doubly counted region
    cross_sum -= np.sum(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    cross_pixels -= np.size(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    # Calculate average
    cross_average = cross_sum/cross_pixels

    # Calculate background std, number of pixels, and average secondary electron yield
    background_left, background_right, background_top, background_bottom = int(pixels_y*1/20), int(pixels_y*8/20), int(pixels_x*1/20), int(pixels_x*8/20)
    background_grid = picture_grid[background_left:background_right, background_top:background_bottom]
    background_sum = np.sum(background_grid)
    background_pixels = np.size(background_grid)
    background_std = np.std(background_grid)
    background_average = background_sum/background_pixels

    # Calculate CNR
    CNR = np.abs(cross_average-background_average)/background_std
    return CNR

def detect_and_plot_harris_corners(
    denoised_grid, 
    block_size=2, 
    ksize=3, 
    k=0.04, 
    threshold_ratio=0.01, 
    dot_radius=1, 
    dot_alpha=0.5,  # Transparency: 0 (invisible) to 1 (solid)
    percentile=50
):
    """
    Perform Harris Corner Detection on a grayscale image and plot it with transparent red dots on corners.

    Parameters:
        denoised_grid (np.ndarray): Grayscale input image as a 2D NumPy array.
        block_size (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter for the Sobel operator.
        k (float): Harris detector free parameter.
        threshold_ratio (float): Threshold relative to max corner response (between 0 and 1).
        dot_radius (int): Radius of the red dots.
        dot_alpha (float): Opacity of the red dots (0 = transparent, 1 = opaque).
    """
    denoised_grid = denoised_grid/np.max(denoised_grid)
    threshold_value = np.max(denoised_grid)*percentile
    # Step 2: Apply thresholding
    denoised_grid = (denoised_grid >= threshold_value).astype(np.uint8)

    # Step 1: Convert to float32 for Harris
    gray = np.float32(denoised_grid)

    # Step 2: Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    dst = cv2.dilate(dst, None)

    # Step 3: Thresholding
    threshold = threshold_ratio * dst.max()
    corner_mask = dst > threshold

    # Step 4: Scale image properly
    if denoised_grid.max() <= 1.0:
        display_img = (denoised_grid * 255).astype(np.uint8)
    else:
        display_img = denoised_grid.astype(np.uint8)

    # Convert grayscale to color
    base_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # Step 5: Create overlay for drawing transparent dots
    overlay = base_img.copy()

    y_coords, x_coords = np.where(corner_mask)
    for x, y in zip(x_coords, y_coords):
        cv2.circle(overlay, (x, y), radius=dot_radius, color=(0, 0, 255), thickness=-1)

    # Step 6: Blend base image with overlay (transparent red dots)
    blended = cv2.addWeighted(overlay, dot_alpha, base_img, 1 - dot_alpha, 0)

    # Step 7: Display
    if show_plots == True:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title("Harris Corners with Transparent Red Dots")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
    return denoised_grid

def cross_position(picture_grid_denoised, percentile):
    # Calculate threshold value based on specified percentile
    threshold_value = np.max(picture_grid_denoised)*percentile
    print(f"Threshold value ({percentile} percentile): {threshold_value:.6f}")
    
    # Find coordinates where array values exceed the threshold
    # np.where returns (y_coordinates, x_coordinates)
    y_coords, x_coords = np.where(picture_grid_denoised > threshold_value)
    
    # Convert to list of (x, y) coordinate tuples
    cross_points = list(zip(x_coords, y_coords))
    
    print(f"Found {len(cross_points)} points above threshold")

    if len(cross_points) > 0:
        points_array = np.array(cross_points)  # shape = (N, 2)
        
        center_x = np.mean(points_array[:, 0])
        center_y = np.mean(points_array[:, 1])
        
        print(f"Center point: x = {center_x:.2f}, y = {center_y:.2f}")
    else:
        print("No points found above threshold.")
    
    return center_x, center_y, cross_points

def find_rotation(img, x, y,cross_length=100e-9,cross_width=14e-9,frame_width_x=1e-6,frame_width_y=1e-6):
    score = np.zeros(90)
    for i in np.arange(0,90):
        img_no_noise = real_image(cross_length=cross_length,cross_line_width=cross_width,shift_x=x,shift_y=y,rotation=i,background_noise=False,frame_width_x=frame_width_x,frame_width_y=frame_width_y)[0]
        # img_no_noise = img_no_noise/np.max(img_no_noise)
        score[i] = np.mean((img-img_no_noise)**2)
    best_i = np.argmin(score)
    angles_refined = np.arange(best_i-2.5,best_i+2.5,0.01)
    score_refined = np.zeros(np.shape(angles_refined))
    for j,angle in enumerate(angles_refined):
        img_no_noise = real_image(cross_length=cross_length,cross_line_width=cross_width,shift_x=x,shift_y=y,rotation=angle,background_noise=False,frame_width_x=frame_width_x,frame_width_y=frame_width_y)[0]
        # img_no_noise = img_no_noise/np.max(img_no_noise)
        score_refined[j] = np.mean((img-img_no_noise)**2)
    best_angle = angles_refined[np.argmin(score_refined)]
    return best_angle







if __name__ == "__main__":

# ===================== Parameters =====================
    # Beam current (in A)
    beam_current = 0.2e-9
    e = 1.60217663e-19 # electron charge (in Coulomb)
    # Fraction of SEs that successfully leave the surface (and will subsequently be detected)
    escape_factor = 0.2
    collector_efficiency = 0.8
    background_noise_level = 0.8 # SE yield of background * 2
    SE_yield_cross = 1 # SE yield without background SE yield
    SE_yield = background_noise_level/2 + SE_yield_cross
    # Scan time per pixel (in s) (inverse of the scan rate)

    SNR = 5
    scan_time_per_pixel = SNR**2/(SE_yield * escape_factor * collector_efficiency * (beam_current/e))
    
    # Pixel size (in m)
    pixel_width_x = 15*1e-9
    pixel_width_y = pixel_width_x
    
    # Pixel size of (real) original image (not really a pixel,since approximates reality) (in m)
    pixel_width_real_x_max = 5*1e-9
    pixel_width_real_y_max = 5*1e-9
    pixel_width_real_x = calculate_original_pixel_size(pixel_width_x,pixel_width_real_x_max)
    pixel_width_real_y = calculate_original_pixel_size(pixel_width_y,pixel_width_real_y_max)

    resize_factor = int(np.round(pixel_width_x/pixel_width_real_x))

    # Frame width (in m)
    frame_width_x = 8*1e-6
    frame_width_y = frame_width_x
    
    # To model beam alignment error, the position of the center of the beam is normally distributed 
    # around the position of the targeted pixel, with standard deviation error_std (in m).
    error_std = 8e-9 # (8e-9 is a guess based on the breakdown of the sources of alignment error)
    
    # Create alignment mark (a cross of high SE yield (background +1 in the middle of the grid)
    # Dimensions in meter
    cross_length = 200e-9
    cross_line_width = 30e-9 # (15e-9 assumed to be critical dimension (CD), i.e. the thinnest line that can be printed)
    
    

    show_plots = True
    rotation_find_boolean = False

    simulation_runs=0
    intensity_threshold=0.95
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
