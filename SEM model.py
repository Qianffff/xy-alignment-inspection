import numpy as np
import matplotlib.pyplot as plt
import scipy
import config as c


"""
The chip and all its features (like edges and alignment marks) are represented with the grid.
The grid has a certain number of x pixels (pixels_x) and y pixels (pixels_y).
Each pixel has a certain value, which is related to how many secondary electrons are released 
when a primary electron hits the chip.

After this a convolution is performed on each pixel and this resulting number is related to
how many secondary electrons are released when the Gaussian shaped beam hits the pixel. 
Note that the Gaussian beam, due to its spread, also has overlap on the surrounding pixels.
The width of the Gaussian distribution is directly related to the probe size of the electron beam. 
The probe size is typically expressed in terms of its full width at half maximum (FWHM)
or FW50, which can be obtained from the beam spot model by considering contributions 
from diffraction, spherical aberration, chromatic aberration, and source size. 

Once this new convolution grid is calculated, we multiply the convoluted grid by the scanning time
as scanning twice as long will result in twice as many secondary electrons. We also multiply by the
intensity of the beam (current of the beam), as doubling the current doubles the expected number of 
secondary electrons.

Now we got a grid of expected number of secondary electrons per pixel. Since only discrete number 
secondary electrons can be released, we model this release of secondary electrons using a Poisson
distribution on each pixel separately (since all pixels have their own number of expected secondary
electrons). The expectation parameter of this Poisson distribution is thus equal to the earlier 
calculated expectation number of secondary electrons.

Using this Poisson distribution, we get a grid of integers, where each integer represents the
number of secondary electrons detected on that pixel.

Note: for low values of measure time per pixel / beam intensity, the effect of shot noise is
significant, and the signal to noise ratio is low. Increasing one of the two (or both) results 
in a higher signal to noise ratio.
"""

# Width of pixel and frame in meter:
pixel_width_x = 2e-9 
pixel_width_y = pixel_width_x
frame_width_x = 1e-6
frame_width_y = frame_width_x


# The number of pixels in the x and the y direction 
pixels_x = int(np.rint(frame_width_x/pixel_width_x))
pixels_y = int(np.rint(frame_width_y/pixel_width_y))

# Can be seen as grid of secondary electron escape factor
# Use the first to have some randomness, and the second for a uniform SE escape factor
grid = np.random.random([pixels_x,pixels_y])
# grid = np.ones([pixels_x,pixels_y])

# Create a cross of high SE escape factor vertically in the middle of the grid.
cross_length_m = 100e-9
cross_line_width = 14e-9
cross_pixel_length = int(cross_length_m/pixel_width_x)
cross_pixel_width = int(cross_line_width/pixel_width_x)

cross_pixel_left_side = int(pixels_x/2 - cross_pixel_length/2)
cross_pixel_right_side = cross_pixel_left_side + cross_pixel_length
cross_pixel_top_side = cross_pixel_left_side
cross_pixel_bottom_side = cross_pixel_right_side

cross_pixel_half_width_plus = int(np.floor(pixels_x/2)+cross_pixel_width/2)
cross_pixel_half_width_minus = int(np.floor(pixels_x/2)-cross_pixel_width/2)


grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus] += 1
# Create a cross of high SE escape factor horizontally in the middle of the grid.
grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side] += 1
# Remove the added value in the intersection of the lines
grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus] -= 1



# Plot the grid of SE escape factors. This is probably how the real chip looks like.
plt.figure(figsize=(13,13))
plt.imshow(grid)
plt.show()


# Defining the function which generates the Gauss kernel which is used in the convolution.
def shifted_gauss1d(gauss_pixels,sigma,shift):
    x = np.arange(gauss_pixels)
    center = (gauss_pixels - 1) / 2 + shift  # shifted center
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def gauss_kernel(gauss_pixels,sigma,shift_x,shift_y):
    gauss1d_x = shifted_gauss1d(gauss_pixels,sigma,shift_x)
    gauss1d_y = shifted_gauss1d(gauss_pixels,sigma,shift_y)
    kernel = np.outer(gauss1d_x, gauss1d_y)
    kernel = kernel / kernel.sum()
    return kernel

# The standard functions in scipy/numpy implemented the convolution function
# to perform the convolution on the entire array, while we want to do it by a 
# per pixel basis since we have a different kernel for each pixel.
# We have a different kernel for each pixel since there is a small error which
# may occur when measuring a pixel.
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
    


# We calculate the expected number of SE per pixel
picture_grid = np.zeros([pixels_x,pixels_y])
expected_number_of_secondary_electrons = np.zeros([pixels_x,pixels_y])


# Some parameters
elementary_charge = 1.60217663 * 10**(-19) # in Coulomb
intensity_beam_A = 1.8*10**(-12) # in Ampere
intensity_beam = intensity_beam_A/(elementary_charge) # number of electrons per second

# Time parameters (4e-7 based on total image time being 0.1 seconds)
scan_time_per_pixel = 6e-6 # in seconds

# The width of the Gaussian is related to the intensity (current) of the beam
FWHM = c.d_p_func(intensity_beam_A)
sigma = FWHM/(2*np.sqrt(2*np.log(2))) # in m
sigma = sigma/pixel_width_x
half_pixel_width_gaussian_kernel = int(np.ceil(3*sigma)) # in pixels

# To store images for each error_m value
# The name 'error_m' stands for error in units of meter.
# The error_m is the error of the beam position. The error from the center of the beam relative to the targetted pixel.
error_m = 8e-9  # in meters


expected_number_of_secondary_electrons = np.zeros((pixels_x, pixels_y))

for i in range(pixels_x):
    for j in range(pixels_y):
        # Random error in pixels in x and y direction
        # The errors are now uniformly chosen, but this can/should probably
        # be changed to a normal distribution with sigma = error_m/2?
        error_shift_x = np.random.uniform(-error_m, error_m) / pixel_width_x
        error_shift_y = np.random.uniform(-error_m, error_m) / pixel_width_y
        
        error_shift_x = np.random.normal(0, error_m) / pixel_width_x
        error_shift_y = np.random.normal(0, error_m) / pixel_width_y
        
        # Each pixel has its own kernel which has its own error. This way
        # the error is new for each pixel. We can add an error which is 
        # dependent on its neighbouring errors.
        kernel_ij = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
                                 sigma,error_shift_x,error_shift_y)
        
        expected_number_of_secondary_electrons[i, j] = convolve_at_pixel(
            grid, kernel_ij, i, j)
    # Progress bar
    if int(np.round(i/pixels_x*100000)) % 5000 == 0:
        print(str(int(np.round(i/pixels_x*100)))+str("%"),end=" ")


# Multiply by intensity and scan time
expected_number_of_secondary_electrons *= intensity_beam * scan_time_per_pixel

# Simulate detected electrons using Poisson statistics
picture_grid = np.random.poisson(expected_number_of_secondary_electrons)

# Plotting
plt.figure(figsize=(12,12))
plt.imshow(picture_grid)
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.savefig("Image result.svg")
plt.show()

time_to_make_picture = pixels_x*pixels_y*scan_time_per_pixel
print(f"Time to make image = {time_to_make_picture:.5f} seconds")
print(f"Scan time per pixel = {scan_time_per_pixel*10**6:.5f} microseconds")
print(f"Beam current = {intensity_beam_A*10**12} pA")
print(f"Error = {error_m*10**9:.3f} nm")


# Contrast to noise ratio

# Cross pixels
cross_sum = 0
cross_pixels = 0

cross_sum += np.sum(picture_grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
cross_pixels += np.size(picture_grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
# Create a cross of high SE escape factor horizontally in the middle of the grid.
cross_sum += np.sum(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side]) 
cross_pixels += np.size(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side])
# Remove the added value in the intersection of the lines
cross_sum -= np.sum(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
cross_pixels -= np.size(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
cross_average = cross_sum/cross_pixels

# Background pixels
background_left, background_right, background_top, background_bottom = int(pixels_y*1/20), int(pixels_y*8/20), int(pixels_x*1/20), int(pixels_x*8/20)
background_grid = picture_grid[background_left:background_right, background_top:background_bottom]
background_sum = np.sum(background_grid)
background_pixels = np.size(background_grid)
background_std = np.std(background_grid)
background_average = background_sum/background_pixels

# Contrast to noise ratio
CNR = np.abs(cross_average-background_average)/background_std
print(f"Contrast to noise ratio = {CNR}")

# Making the beam intenisty vs time plot. (Using CNR test CNR = 2)
scan_time_per_image_array = np.array([0.1,0.11,0.11375,0.115,0.1175,0.1195,0.12125,0.1225,0.1275,0.13,0.13375,0.1475,0.16375,0.18,0.19250,0.19250,0.20000,0.19750,0.22500,0.25250,0.27625,0.30125,0.32750,0.35375,0.39250,0.50000,0.60000,0.77500,0.95000,1.50000,2.95000])
scan_time_per_pixel_array = scan_time_per_image_array/(pixels_x*pixels_y)
beam_intensity_A_array = np.array([3.1,3,2.9,2.8,2.7,2.6,2.5,2.4,2.3,2.2,2.1,2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])*10**(-12)


plt.figure()
plt.plot(beam_intensity_A_array*10**12,scan_time_per_image_array,"k.-")
plt.xlabel("Beam current (pA)")
plt.ylabel("Time per 1 µm² image (s)")
plt.savefig("I vs time.svg")
plt.show()
