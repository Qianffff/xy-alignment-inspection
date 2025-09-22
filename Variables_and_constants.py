import numpy as np

# ===================== Parameters =====================

frame_width = 1*1e-6 # Frame width (in m)
pixel_width = 20*1e-9 # Pixel size (in m)
SNR = 10 # Desired signal to noise ratio

# ============ Simulate alignment procedure ============
# An alignment procedure is a sequence of scans that determine the mark position with increasing accuracy.
# For each step of the alignment procedure, define the frame_width (in m), pixel_width (in m), and desired SNR.
# The parameter 'frame_width' defined above should match the framewidth of the first step of the procedure.
# Format: [frame_width,pixel_width,SNR]
step1 = [1*1e-6,10*1e-9,10]
step2 = [0.5*1e-6,2*1e-9,10]
procedure = [step1,step2]



# ===================== Constants =====================

intensity_threshold=0.8 # Used for finding the cross position
beam_current = 0.5e-9 # Beam current (in A)

e = 1.60217663e-19 # electron charge (in Coulomb)

escape_factor = 0.2 # Fraction of SEs that successfully leave the surface
collector_efficiency = 0.8
average_SE_yield_background = 0.4 # average SE yield of the background
SE_yield_cross_edge = 1 # SE yield of the cross edges (excluding background)
SE_yield_cross_body = 0.5 # SE yield of the cross body (excluding background)
SE_yield = SE_yield_cross_edge + average_SE_yield_background # Only used to calculate the scan time per pixel

FWHM = 8e-9 # Full width half maximum of the electron beam (in m)

# To model beam alignment error, the position of the center of the beam is normally distributed 
# around the position of the targeted pixel, with standard deviation error_std (in m).
beam_placement_error_std = 8*1e-9 # (8*1e-9 is a guess based on the breakdown of the sources of alignment error)

drift_rate = 5*1e-9/60 # Beam drift rate (in m/s).
FOV_count = 1 # (Minimum is 1.) Number of FOVs the beam has scanned already. Used to calculate how much beam drift has accumulated.

# Create alignment mark (a cross of high SE yield (background +1 in the middle of the grid)
# Dimensions in meter
cross_length =  200*1e-9
cross_linewidth = 30*1e-9 # (15e-9 assumed to be critical dimension (CD), i.e. the thinnest line that can be printed)

# Pixel size of real image (not really a pixel, since it approximates reality) (in m)
pixel_width_real = 1*1e-9

# Time it takes per unit distance the beam has to go from one side of its FOV to the other to scan a new row of pixels, per meter it has to traverse (in s/m)
beam_overhead_rate = 0.1
# Time it takes to send SEM data to a computer, process it (i.e. find the mark), and send the result back to the SEM
latency = (0.1 + 1 + 0.1)*1e-3

# Optical microscope FOV area (in m)
ebeam_FOV_width = 8*1e-6
optical_microscope_FOV_width = 10*1e-6

a = 2*9.81 # Maximum stage accelaration (in m/s²)

# ===================== Derived parameters =====================

# Total time spent moving the stage to go from one row of FOV images to the next in the first step of the alignment procedure
stage_overhead_time = 2*np.sqrt(2/a*(ebeam_FOV_width/2)) * (np.ceil(optical_microscope_FOV_width/ebeam_FOV_width)-1)

# Number of ebeam FOVs needed to cover the optical microscope FOV
n_eFOVs = (np.ceil(optical_microscope_FOV_width/ebeam_FOV_width))**2
# Scan time per pixel (in s)
scan_time_per_pixel = SNR**2/(SE_yield * escape_factor * collector_efficiency * (beam_current/e)) # Scan time per pixel (in s) (inverse of the scan rate)

# Number of pixels (in both x and y) of the real wafer
pixels_real = int(frame_width/pixel_width_real)
if pixels_real % 2 == 0: pixels_real +=1

# Number of pixels (in both x and y) of the measured image
pixels = int(frame_width/pixel_width)
if pixels % 2 == 0: pixels += 1

# Calculate the beam width given the beam current
sigma = FWHM/(2*np.sqrt(2*np.log(2))) # (in m)
sigma = sigma/pixel_width # (in px)
kernel_width = int(2 * np.ceil(3*sigma) + 1) # (in px)

show_plots = True

simulation_runs=0 # For SEM simulation histogram