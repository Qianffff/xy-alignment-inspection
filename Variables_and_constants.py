import numpy as np

def calculate_original_pixel_size(new_pixel_size_nm,original_pixel_size_nm_maximum):
    original_pixel_size_nm = new_pixel_size_nm
    while original_pixel_size_nm > original_pixel_size_nm_maximum:
        original_pixel_size_nm = original_pixel_size_nm/2
    return original_pixel_size_nm


# ===================== Parameters =====================
# Beam current (in A)
beam_current = 0.5e-9
e = 1.60217663e-19 # electron charge (in Coulomb)
# Fraction of SEs that successfully leave the surface (and will subsequently be detected)
escape_factor = 0.2
collector_efficiency = 0.8
background_noise_level = 0.8 # SE yield of background * 2
SE_yield_cross = 1 # SE yield without background SE yield
SE_yield = background_noise_level/2 + SE_yield_cross
# Scan time per pixel (in s) (inverse of the scan rate)

SNR = 10
scan_time_per_pixel = SNR**2/(SE_yield * escape_factor * collector_efficiency * (beam_current/e))

# Pixel size (in m)
pixel_width_x = 5*1e-9
pixel_width_y = pixel_width_x

# Pixel size of (real) original image (not really a pixel,since approximates reality) (in m)
pixel_width_real_x_max = 1*1e-9
pixel_width_real_y_max = 1*1e-9
pixel_width_real_x = calculate_original_pixel_size(pixel_width_x,pixel_width_real_x_max)
pixel_width_real_y = calculate_original_pixel_size(pixel_width_y,pixel_width_real_y_max)

resize_factor = int(np.round(pixel_width_x/pixel_width_real_x))

# Frame width (in m)
frame_width_x = 1*1e-6
frame_width_y = frame_width_x

pixels_x = int(np.rint(frame_width_x/pixel_width_real_x))
pixels_y = int(np.rint(frame_width_y/pixel_width_real_y))

if pixels_x % 2 == 0:
    pixels_x += 1

if pixels_y % 2 == 0:
    pixels_y += 1

# Calculate the beam width given the beam current
FWHM = 8e-9 # (in m)
sigma = FWHM/(2*np.sqrt(2*np.log(2))) # (in m)
sigma = sigma/pixel_width_x # (in px)
half_pixel_width_gaussian_kernel = int(np.ceil(3*sigma)) # (in px)

# To model beam alignment error, the position of the center of the beam is normally distributed 
# around the position of the targeted pixel, with standard deviation error_std (in m).
error_std = 8e-9 # (8e-9 is a guess based on the breakdown of the sources of alignment error)

# We expect position error due to max allowable pos. error of 8nm in the time it takes to make one grid (about 60 seconds).
drift_rate = 5e-9/60 # m/s position error due to drift. 
FOV_count = 10000 # needs to be >= 1

# Create alignment mark (a cross of high SE yield (background +1 in the middle of the grid)
# Dimensions in meter
cross_length = 200e-9
cross_line_width = 30e-9 # (15e-9 assumed to be critical dimension (CD), i.e. the thinnest line that can be printed)



show_plots = True
rotation_find_boolean = False

simulation_runs=0
intensity_threshold=0.7