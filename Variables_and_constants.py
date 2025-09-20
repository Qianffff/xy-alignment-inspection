import numpy as np

# ===================== Parameters =====================

frame_width = 1*1e-6 # Frame width (in m)
pixel_width = 20*1e-9 # Pixel size (in m)
SNR = 10 # Desired signal to noise ratio

# ============ Simulate alignment procedure ============
# An alignment procedure is a sequence of scans that determine the mark position with increasing accuracy.
# For each step of the alignment procedure, define the frame_width (in m), pixel_width (in m), and desired SNR.
# Format: [frame_width,pixel_width,SNR]
step1 = [1e-6,20e-9,5]
step2 = [0.5e-6,5e-9,10]
procedure = [step1,step2]



# ===================== Constants =====================

intensity_threshold=0.7 # Used for finding the cross position
beam_current = 0.5e-9 # Beam current (in A)
e = 1.60217663e-19 # electron charge (in Coulomb)
escape_factor = 0.2 # Fraction of SEs that successfully leave the surface
collector_efficiency = 0.8
background_noise_level = 0.8 # SE yield of background * 2
SE_yield_cross = 1 # SE yield without background SE yield
SE_yield = background_noise_level/2 + SE_yield_cross
FWHM = 8e-9 # Full width half maximum of the electron beam (in m)

# To model beam alignment error, the position of the center of the beam is normally distributed 
# around the position of the targeted pixel, with standard deviation error_std (in m).
error_std = 8e-9 # (8e-9 is a guess based on the breakdown of the sources of alignment error)

drift_rate = 5*1e-9/60 # Beam drift rate (in m/s).
FOV_count = 1 # (Minimum is 1.) Number of FOVs the beam has scanned already. Used to calculate how much beam drift has accumulated.

# Create alignment mark (a cross of high SE yield (background +1 in the middle of the grid)
# Dimensions in meter
cross_length = 200e-9
cross_line_width = 30e-9 # (15e-9 assumed to be critical dimension (CD), i.e. the thinnest line that can be printed)

# Pixel size of real image (not really a pixel, since it approximates reality) (in m)

pixel_width_real = 1e-9
# Is this necessary?
# pixel_width_real_max = 1e-9
# pixel_width_real = pixel_width
# while pixel_width_real > pixel_width_real_max:
#     pixel_width_real = pixel_width_real/2



# ===================== Derived parameters =====================

scan_time_per_pixel = SNR**2/(SE_yield * escape_factor * collector_efficiency * (beam_current/e)) # Scan time per pixel (in s) (inverse of the scan rate)

pixels_real = int(np.round(frame_width/pixel_width_real)) # Number of pixels (in both x and y) of the real wafer

pixels = int(np.round(frame_width/pixel_width)) # Number of pixels (in both x and y) of the measured image
if pixels % 2 == 0: pixels += 1



# Calculate the beam width given the beam current
sigma = FWHM/(2*np.sqrt(2*np.log(2))) # (in m)
sigma = sigma/pixel_width # (in px)
half_pixel_width_gaussian_kernel = int(np.ceil(3*sigma)) # (in px)

show_plots = True

simulation_runs=0