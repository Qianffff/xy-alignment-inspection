import math
import numpy as np

# Calculate the time of an alignment procedure
def show_time(procedure):
    total_time = 0
    time_list = []
    for mark in procedure:
        for step in mark:
            n, pixel_width, SNR = step # Get the parameters that define the step
            
            # Calculate intermediate parameters used to find the time
            scan_time_per_pixel = SNR**2/(SE_yield*SE_escape_factor*collector_efficiency * (beam_current/e))
            pixels = np.sqrt(FOV_area) / pixel_width
            
            FOV_time = scan_time_per_pixel*pixels**2 + beam_overhead_rate*(pixels*pixel_width)*(pixels-1) # Time of one FOV
            step_time = FOV_time * n + stage_overhead_time_alignment_per_move * n + latency # Time of the step
            total_time += step_time # Update total time of the procedure
            
            time_list.append(step_time)
    return total_time, time_list


###################### Machine-speficic parameters #############################


beam_number_1100 = 25
beam_current_1100 = 4e-9 # A
beam_pitch_1100 = 8e-6 # m
FOV_area_1100 = 64e-12 # m²
n_align_per_grid_1100 = 0.01

beam_number_2200 = 2791
beam_current_2200 = 0.5e-9 # A
beam_pitch_2200 = 100e-6 # m
FOV_area_2200 = 1e-12 # m²
n_align_per_grid_2200 = 1


##################### General parameters ########################


optical_microscope_accuracy = 10*1e-6 # m
SNR_inspection = 10 # Signal to noise ratio during inspection
SE_yield = 1
SE_escape_factor = 0.2                     
collector_efficiency = 0.8
cross_length = 1*1e-6 # m
a = 2 * 9.81 # Maximum stage acceleration (m/s²)
pixel_width = 5e-9 # m
beam_overhead_rate = 0.1 # s/m
latency = (0.1 + 1 + 0.1)*1e-3 # s

# Constants
e = 1.602e-19 # C


#################### Define alignment procedures ###########################


# An alignment procedure is built up like this:
#
# procedure = [mark_1,mark_2,mark_3,...] (We currently always use 3 marks)
# mark_i = [step_i_1,step_i_2,...] (We currently always use 2 steps per mark)
# step_i.j = [n,pixel_width,SNR]

# Create alignment procedure for the 2200:

n_min = (2*cross_length)**2 / FOV_area_2200 # Minimum number of FOVs needed to image the full cross (with some margin)

step_1_1 = [300, 20e-9, 5]
step_1_2 = [n_min, 5e-9, 10]
mark_1 = [step_1_1,step_1_2]

step_2_1 = [30, 20e-9, 5]
step_2_2 = [n_min, 5e-9, 10]
mark_2 = [step_2_1, step_2_2]

step_3_1 = [n_min, 5e-9, 10]
mark_3 = [step_3_1]

procedure2200 = [mark_1,mark_2,mark_3]

# Create alignment procedure for the 1100:

n_min = (2*cross_length)**2 / FOV_area_1100 # Minimum number of FOVs needed to image the full cross (with some margin)

step_1_1 = [5, 20e-9, 5]
step_1_2 = [n_min, 5e-9, 10]
mark_1 = [step_1_1,step_1_2]

step_2_1 = [n_min, 5e-9, 10]
mark_2 = [step_2_1]

step_3_1 = [n_min, 5e-9, 10]
mark_3 = [step_3_1]

procedure1100 = [mark_1,mark_2,mark_3]


################## Quick switch between machine settings #########################


# [beam_number , beam_current , beam_pitch , FOV_area, n_realign_per_grid]
settings1100 = [beam_number_1100, beam_current_1100, beam_pitch_1100, FOV_area_1100, n_align_per_grid_1100, procedure1100]
settings2200 = [beam_number_2200, beam_current_2200, beam_pitch_2200, FOV_area_2200, n_align_per_grid_2200, procedure2200]
settings_test = [0,0,0,0,0]

beam_number, beam_current, beam_pitch, FOV_area, n_realign_per_grid, procedure = settings2200


############################# Calculations ####################################


pixels = int(np.sqrt(FOV_area)/pixel_width)

N_SE_required = SNR_inspection**2 # number of detected SEs to make image
stage_overhead_time_alignment_per_move = 2*np.sqrt(2/a*(np.sqrt(FOV_area/2))) 
stage_overhead_time_per_grid = 2*np.sqrt(2/a*(np.sqrt(FOV_area/2))) * (np.ceil(beam_pitch/np.sqrt(FOV_area))-1)

# Pixel scan time
pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ collector_efficiency) / (beam_current/e)

# Number of FOV images needed to find mark
FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1) # s

# Alignment time
initial_alignment_time, time_list = show_time(procedure)
realignment_time = time_list[-1]*3 # Factor 3 for the three marks

# Beam scan rate (per beam)
beam_scan_rate = FOV_area / FOV_scan_time

# Total scan rate (all beams)
scan_rate = beam_number * beam_scan_rate

# Grid area
grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number # sqrt(3/4) due to hexagonal grid shape

# Grid scan time
grid_scan_time = grid_area / scan_rate + stage_overhead_time_per_grid

# Pixel area (approx)
pixels_per_FOV = FOV_scan_time / pixel_scan_time
pixel_area = pixel_width**2

n_eFOVs_to_align = 0
for mark in procedure:
    for step in mark:
        n_eFOVs_to_align += step[0]

scan_time_per_alignement = grid_scan_time / n_realign_per_grid
scanning_fraction = scan_time_per_alignement / (realignment_time + scan_time_per_alignement)
effective_throughput = scan_rate * scanning_fraction
absolute_throughput_loss = scan_rate - effective_throughput
relative_troughpit_loss = absolute_throughput_loss/scan_rate


###################### Print numbers #########################

# print(f"Pixel area = {pixel_area*1e18:.0f} nm²")
print(f"Pixel scan time = {pixel_scan_time*1e9:.2f} ns")

print(f"Number of pixels in FOV = {pixels**2:.0f}")
print(f"FOV scan time = {FOV_scan_time:.4f} s")

print(f"Grid area = {grid_area*1e6:.5f} mm²")
print(f"Grid scan time = {grid_scan_time:.5f} s")

print(f"Number of FOV images for initial alignment = {n_eFOVs_to_align:.0f}")
print(f"Initial alignment time = {initial_alignment_time:.4f} s")

# print(f"SNR during inspection = {SNR_inspection:.0f}")
print(f"Stage overhead time during initial alignment = {stage_overhead_time_alignment_per_move*n_eFOVs_to_align:.6f} s")
print(f"Stage overhead time per inspected grid = {stage_overhead_time_per_grid:.6f} s")

# print(f"Beam scan rate = {beam_scan_rate*1e12:.0f} µm²/s")
print(f"Scan rate = {scan_rate*1e6*3600:.1f} mm²/h")

# Analysis of throughput losses
print("#################################################################")

print(f"Scan time per alignment = {scan_time_per_alignement:.3f} s")
print(f"Realignment time = {realignment_time:.5f} s")
print(f"Scanning fraction = {scanning_fraction:.5f}")
print(f"Effective throughput = {effective_throughput*1e6*3600:.2f} mm²/h")
print(f"Absolute throughput loss = {absolute_throughput_loss*1e6*3600:.2f} mm²/h")
print(f"Relative throughput loss = {relative_troughpit_loss*100:.3f} %")