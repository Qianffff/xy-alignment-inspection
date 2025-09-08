import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal.windows import gaussian
from scipy.signal import convolve2d
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

# Width of pixel and frame in nanometer:
pixel_width_x = 2e-9 
pixel_width_y = pixel_width_x
frame_width_x = 1e-6 # = 1000 nm
frame_width_y = frame_width_x


# The number of pixels in the x and the y direction 
pixels_x = int(np.rint(frame_width_x/pixel_width_x))
pixels_y = int(np.rint(frame_width_y/pixel_width_y))

# Can be seen as grid of secondary electron escape factor
# Use the first to have some randomness, and the second for a uniform SE escape factor
grid = np.random.random([pixels_x,pixels_y])/2
# grid = np.ones([pixels_x,pixels_y])*0.1

# Create a line of high SE escape factor vertically in the middle of the grid.
grid[:,int(np.floor(pixels_x/2))] += 1
grid[int(np.floor(pixels_y/2)),:] += 1
grid[int(np.floor(pixels_y/2)),int(np.floor(pixels_x/2))] -= 1


grid *= 1
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

# Show the Gauss kernel 
kernel = gauss_kernel(310,50,0,100)
plt.figure(figsize=(13,13))
plt.imshow(kernel)
plt.show()




# NOTE: Below we used a Gaussian filter, which is not realistic since the result contains
# non integer numbers for the escaped electrons, which in reality needs to be an integer.

# Apply the Gauss filter
sigma = 2
picture_grid = scipy.ndimage.gaussian_filter(grid,sigma)

# Plot the result of the Gauss filter
plt.figure(figsize=(13,13))
plt.imshow(picture_grid)
plt.show()

# Now the more realistic approach is used, where we first calculate the expected 
# number of SE per pixel
picture_grid = np.zeros([pixels_x,pixels_y])
expected_number_of_secondary_electrons = np.zeros([pixels_x,pixels_y])


# Some parameters
elementary_charge = 1.60217663 * 10**(-19) # in Coulomb
intensity_beam_A = 570*10**(-12)/2 # in Ampere
intensity_beam = intensity_beam_A/(elementary_charge) # number of electrons per second

# Time parameters

scan_time_per_pixel = 1/(20*10**6) # in seconds

# The width of the Gaussian is related to the intensity (current) of the beam
FWHM = c.d_p_func(intensity_beam_A)
sigma = FWHM/(2*np.sqrt(2*np.log(2))) # in m
sigma = sigma/pixel_width_x
half_pixel_width_gaussian_kernel = int(np.ceil(3*sigma)) # in pixels

# To store images for each error_m value
picture_grids = []
error_values = [0, 5e-9]  # in meters

for error_m in error_values:
    expected_number_of_secondary_electrons = np.zeros((pixels_x, pixels_y))

    for i in range(pixels_x):
        for j in range(pixels_y):
            # Random error in pixels in x and y direction
            error_shift_x = np.random.uniform(-error_m, error_m) / pixel_width_x
            error_shift_y = np.random.uniform(-error_m, error_m) / pixel_width_y

            kernel_ij = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
                                     sigma,error_shift_x,error_shift_y)

            expected_number_of_secondary_electrons[i, j] = convolve2d(
                grid,
                kernel_ij,
                mode='same',
                boundary='fill',
                fillvalue=0.5)[i, j]

    # Multiply by intensity and scan time
    expected_number_of_secondary_electrons *= intensity_beam * scan_time_per_pixel

    # Simulate detected electrons using Poisson statistics
    picture_grid = np.random.poisson(expected_number_of_secondary_electrons)

    # Store the image
    picture_grids.append(picture_grid)

# Plotting both images side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for idx, ax in enumerate(axs):
    im = ax.imshow(picture_grids[idx], cmap='viridis')
    ax.set_title(f"Error = {error_values[idx]*1e9:.1f} nm")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

time_to_make_picture = pixels_x*pixels_y*scan_time_per_pixel
print(f"Time to make image = {time_to_make_picture:.5f} seconds")

# # Show pictures for different values of scan time per pixel
# for scan_time_per_pixel in [10**(-6)*1,10**(-6)*100,10**(-6)*10000,10**(-6)*100000]:
#     for i in range(pixels_x):
#         for j in range(pixels_y):
#             # Random error in pixels in x and y direction
#             error_shift_x = np.random.uniform(-error_nm,error_nm)/pixel_width_x
#             error_shift_y = np.random.uniform(-error_nm,error_nm)/pixel_width_y
            
#             kernel_ij = gauss_kernel(2*half_pixel_width_gaussian_kernel+1,
#                                      sigma,error_shift_x,error_shift_y)
#             expected_number_of_secondary_electrons[i,j] = convolve2d(grid, 
#                                                                 kernel_ij,
#                                                                 mode='same', 
#                                                                 boundary='fill', 
#                                                                 fillvalue=0.5)[i,j] 
#     expected_number_of_secondary_electrons *= intensity_beam * scan_time_per_pixel
#     picture_grid = np.random.poisson(expected_number_of_secondary_electrons)
       
#     # Plot the realistic approach which used the Poisson distribution     
#     plt.figure(figsize=(13,13))
#     plt.title(f"Time per pixel = {np.round(scan_time_per_pixel*1000000)} microseconds")
#     plt.imshow(picture_grid)
#     plt.colorbar()
#     plt.show()