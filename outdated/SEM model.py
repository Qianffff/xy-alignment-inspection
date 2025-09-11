import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import config as c
from skimage.restoration import denoise_nl_means, estimate_sigma

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
def real_image(pixel_width_x=2e-9,pixel_width_y=2e-9,frame_width_x=1e-6,frame_width_y=1e-6,cross_length=100e-9,cross_line_width=14e-9):
    # The number of pixels in the x and the y direction 
    pixels_x = int(np.rint(frame_width_x/pixel_width_x))
    pixels_y = int(np.rint(frame_width_y/pixel_width_y))
    
    # Grid of secondary electron (SE) escape factors
    grid = np.zeros([pixels_x,pixels_y])
    
    # Dimensions in pixels
    cross_pixel_length = int(np.round(cross_length/pixel_width_x))
    cross_pixel_width = int(np.round(cross_line_width/pixel_width_x))

    # Define some useful parameters about the cross geometry
    cross_pixel_left_side = int(np.round(pixels_x/2 - cross_pixel_length/2))
    cross_pixel_right_side = cross_pixel_left_side + cross_pixel_length
    cross_pixel_top_side = cross_pixel_left_side
    cross_pixel_bottom_side = cross_pixel_right_side
    cross_pixel_half_width_plus = int(np.round(pixels_x/2+cross_pixel_width/2))
    cross_pixel_half_width_minus = int(np.round(pixels_x/2-cross_pixel_width/2))

    # Random position and rotation cross
    max_shift_x = int(np.round(cross_pixel_left_side - 1/8*pixels_x))
    max_shift_y = int(np.round(cross_pixel_top_side - 1/8*pixels_y))
    shift_x = int(np.random.randint(-max_shift_x,max_shift_x))
    shift_y = int(np.random.randint(-max_shift_y,max_shift_y))
    rotation = np.random.uniform(0,90)

    # First: create the cross in the middle of the grid
    # Create the vertical line
    grid[cross_pixel_top_side:cross_pixel_bottom_side,
         cross_pixel_half_width_minus:cross_pixel_half_width_plus] += 1
    # Create the horizontal line
    grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,
         cross_pixel_left_side:cross_pixel_right_side] += 1
    # Remove doubly counted region
    grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,
         cross_pixel_half_width_minus:cross_pixel_half_width_plus] -= 1

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
    # Use the first to have some randomness, and the second for a constant escape factor
    grid += np.random.random([pixels_x,pixels_y])
    # grid = np.ones([pixels_x,pixels_y])
    return grid, pixels_x, pixels_y, shift_x, shift_y, rotation

def measured_image(real_image,beam_current=500e-12,scan_time_per_pixel=4e-7,error_std=8e-9):
    # Calculate the expected number of SE per pixel
    pixels_x = np.shape(real_image)[0]
    pixels_y = np.shape(real_image)[0]
    picture_grid = np.zeros((pixels_x, pixels_y))
    expected_number_of_secondary_electrons = np.zeros((pixels_x, pixels_y))
    
    for i in range(pixels_x):
        for j in range(pixels_y):
            # Random beam position error in x and y direction (in pixels)
            error_shift_x = np.random.normal(scale=error_std) / pixel_width_x
            error_shift_y = np.random.normal(scale=error_std) / pixel_width_y
            # Create kernel
            kernel_ij = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
                                     sigma,error_shift_x,error_shift_y)
            # Perform the convolution
            expected_number_of_secondary_electrons[i, j] = convolve_at_pixel(
                grid, kernel_ij, i, j)
        # Progress bar
        if int(np.round(i/pixels_x*100000)) % 5000 == 0:
            print(str(int(np.round(i/pixels_x*100)))+str("%"),end=" ")

    # Multiply by number of primary electrons per second (= beam current / e) and scan time
    e = 1.60217663e-19 # electron charge (in Coulomb)
    expected_number_of_secondary_electrons *= beam_current/e * scan_time_per_pixel
    # If there is no background noise, some numbers may become smaller than 0.
    # This gives an error in the upcoming Poisson function
    expected_number_of_secondary_electrons[expected_number_of_secondary_electrons<0] = 0


    # Simulate detected electrons using Poisson statistics
    picture_grid = np.random.poisson(expected_number_of_secondary_electrons)
    return picture_grid

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
    plt.show()
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

    cross_sum = 0 # Sum of secondary electron escape factors of all pixels in the cross
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

    # Calculate background std, number of pixels, and average secondary electron escape factor
    background_left, background_right, background_top, background_bottom = int(pixels_y*1/20), int(pixels_y*8/20), int(pixels_x*1/20), int(pixels_x*8/20)
    background_grid = picture_grid[background_left:background_right, background_top:background_bottom]
    background_sum = np.sum(background_grid)
    background_pixels = np.size(background_grid)
    background_std = np.std(background_grid)
    background_average = background_sum/background_pixels

    # Calculate CNR
    CNR = np.abs(cross_average-background_average)/background_std
    return CNR


if __name__ == "__main__":

    # Beam current (in A)
    beam_current = 5e-13 # 570e-12 based on Zeiss specs sheet
    # Scan time per pixel (inverse of the scan rate)
    scan_time_per_pixel = 4e-7 # (in s) (4e-7 based on total image time of 1 um^2 with 2 nm pixel size being 0.1 seconds (the 0.1 s is according to Koen))
    
    # Pixel size (in m)
    pixel_width_x = 2e-9 # (2e-9 is a guess based on the ASML metrology and inspection systems webpage)
    pixel_width_y = pixel_width_x
    
    # Frame width (in m)
    frame_width_x = 1e-6 # (1e-6 according to Koen)
    frame_width_y = frame_width_x
    
    # To model beam alignment error, the position of the center of the beam is normally distributed 
    # around the position of the targeted pixel, with standard deviation error_std (in m).
    error_std = 8e-9 # (8e-9 is a guess based on the breakdown of the sources of alignment error)
    
    # Create alignment mark (a cross of high SE escape factor (background +1 in the middle of the grid)
    # Dimensions in meter
    cross_length = 100e-9
    cross_line_width = 14e-9 # (14e-9 assumed to be critical dimension (CD), i.e. the thinnest line that can be printed)
    
    
    
    
    grid, pixels_x, pixels_y, shift_x, shift_y, rotation = real_image(pixel_width_x,pixel_width_y,frame_width_x,frame_width_y,cross_length,cross_line_width)
    
    #Plot the grid of SE escape factors. This represents what the real wafer pattern looks like.
    plt.figure(figsize=(13,13))
    plt.imshow(grid)
    plt.title('Secondary electron escape factor grid')
    plt.colorbar()
    plt.show()
    print(f"Cross middle x pixel = {int(np.round(pixels_x/2+shift_x))}")
    print(f"Cross middle y pixel = {int(np.round(pixels_y/2+shift_y))}")
    print(f"Rotation = {rotation:.3f}")



    # Calculate the beam width given the beam current
    FWHM = c.d_p_func(beam_current) # (in m)
    FWHM = 9e-9 # (in m)
    sigma = FWHM/(2*np.sqrt(2*np.log(2))) # (in m)
    sigma = sigma/pixel_width_x # (in px)
    half_pixel_width_gaussian_kernel = int(np.ceil(3*sigma)) # (in px)


    plot_kernel(half_pixel_width_gaussian_kernel,sigma)

    picture_grid = measured_image(grid, beam_current, scan_time_per_pixel, error_std)

    # Plotting
    plt.figure(figsize=(12,12))
    plt.imshow(picture_grid)
    plt.title('Simulated SEM image')
    plt.colorbar()
    plt.tight_layout()
    plt.show()






    picture_grid_denoised = denoise_image(picture_grid)
    
    # Plotting the denoised
    plt.figure(figsize=(12,12))
    plt.imshow(picture_grid_denoised)
    plt.title('Simulated SEM image denoised')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    
    
    
    time_to_make_picture = pixels_x*pixels_y*scan_time_per_pixel
    print(f"Time to make image = {time_to_make_picture:.5f} seconds")
    print(f"Scan time per pixel = {scan_time_per_pixel*1e6:.5f} µs")
    print(f"Beam current = {beam_current*1e12} pA")
    print(f"Error std = {error_std*1e9:.3f} nm")





    # CNR = calculate_CNR()
    # print(f"Contrast to noise ratio = {CNR}")
    
    # Make a beam intensity vs time plot of the curve with CNR = 2
    
    # Values of beam current (in pA)
    beam_current_array = np.array([3.1,3,2.9,2.8,2.7,2.6,2.5,2.4,2.3,2.2,2.1,2,
                                   1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,
                                   0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])
    # Values of the time to make the image (in s)
    scan_time_per_image_array = np.array([0.1,0.11,0.11375,0.115,0.1175,0.1195,0.12125,
                                          0.1225,0.1275,0.13,0.13375,0.1475,0.16375,
                                          0.18,0.19250,0.19250,0.20000,0.19750,0.22500,
                                          0.25250,0.27625,0.30125,0.32750,0.35375,0.39250,
                                          0.50000,0.60000,0.77500,0.95000,1.50000,2.95000])
    scan_time_per_pixel_array = scan_time_per_image_array/(pixels_x*pixels_y)
    
    # Make the plot
    plt.figure()
    plt.plot(beam_current_array,scan_time_per_image_array,"k.-")
    plt.xlabel("Beam current (pA)")
    plt.ylabel("Time per 1 µm² image (s)")
    plt.show()