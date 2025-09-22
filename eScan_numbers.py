import math
import numpy as np

# ----------------------------
# Given / assumed parameters
# ----------------------------
beam_number = 25               # number of beams
beam_current = 4e-9            # A 
beam_pitch = 8e-6              # m 
FOV_area = 64e-12                  # m² 
Expected_n_FOV_tofindmark = 4

# Following variables are the same for 1100 and 2200
optical_accuracy = 10e-6       # m
SNR = 10                       # signal to noise ratio
SE_escape_factor = 0.2         
SE_yield = 1                   
Collector_efficiency = 0.8
a = 2 * 9.81                    # m/s**2
pixel_width = 5e-9              # m
beam_overhead_rate = 0.1           # s/m
latency = (0.1 + 1 + 0.1)*1e-3      #s

# Constants
e_charge = 1.602e-19           # C

# ----------------------------
# Calculations
# ----------------------------
pixels = int(np.sqrt(FOV_area)/pixel_width)
if pixels % 2 == 0: pixels += 1

N_SE_required = SNR**2         # number of detected SEs to make image
stage_overhead_time_alignment = 2*np.sqrt(2/a*(np.sqrt(FOV_area/2))) * Expected_n_FOV_tofindmark
stage_overhead_time_inspection = 2*np.sqrt(2/a*(np.sqrt(FOV_area/2))) * (np.ceil(beam_pitch/np.sqrt(FOV_area))-1)



# Pixel scan time
pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ Collector_efficiency) / (beam_current / e_charge)

# Number of FOV images needed to find mark
num_FOV_images = Expected_n_FOV_tofindmark
FOV_scan_time = pixel_scan_time * pixels**2+beam_overhead_rate*pixels*pixel_width           # s

# Alignment time
alignment_time = FOV_scan_time * num_FOV_images + stage_overhead_time_alignment + latency

# Beam scan rate (per beam)
beam_scan_rate = FOV_area / FOV_scan_time

# Total scan rate (all beams)
scan_rate = beam_number * beam_scan_rate

# Grid area
grid_area = (beam_pitch**2) * beam_number

# Grid scan time
grid_scan_time = grid_area / scan_rate + stage_overhead_time_inspection

# Pixel area (approx)
pixels_per_FOV = FOV_scan_time / pixel_scan_time
pixel_area = FOV_area / pixels_per_FOV


print(f"FOV scan time = {FOV_scan_time:.3f} s")
print(f"Number of detected SEs to make image = {N_SE_required:.0f}")
print(f"Stage overhead time for alignment = {stage_overhead_time_alignment:.3f} s")
print(f"Stage overhead time for inspection = {stage_overhead_time_inspection:.3f} s")
print(f"Number of FOV images to find mark = {Expected_n_FOV_tofindmark:.0f}")
print(f"Alignment time = {alignment_time:.8f} s (for one mark)")
print(f"Beam scan rate = {beam_scan_rate*1e12:.0f} µm²/s")
print(f"Scan rate = {scan_rate*1e6*3600:.1f} mm²/h")
print(f"Grid area = {grid_area*1e6:.5f} mm²")
print(f"Grid scan time = {grid_scan_time/60:.5f} minutes")
print(f"Pixel scan time = {pixel_scan_time*1e9:.3f} ns")
print(f"Pixel area = {pixel_area*1e18:.3f} nm²")
print(f"Number of pixels = {pixels:.0f}")