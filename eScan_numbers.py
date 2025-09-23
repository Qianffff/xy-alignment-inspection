import math
import numpy as np
from Variables_and_constants import procedure 

def show_time(procedure,n_eFOVs): 
    total_time = 0
    time_list = []
    for n_eFOV in n_eFOVs:
        for i in range(len(procedure)):
            frame_width, pixel_width, SNR = procedure[i]

            scan_time_per_pixel = SNR**2/(SE_yield*SE_escape_factor*Collector_efficiency * (beam_current/e_charge))

            pixels = frame_width / pixel_width

            time = scan_time_per_pixel*pixels**2 + beam_overhead_rate*(pixels*pixel_width)*(pixels-1)

            if i == 0: time = time * n_eFOV * (np.sqrt(FOV_area)**2/frame_width**2) + stage_overhead_time_alignment_per_move*n_eFOV + latency # Account for taking multiple ebeam FOV sized images in the first step to find the mark
            else: time += latency
            total_time += time
            time_list.append(time)
    return total_time, time_list


# ----------------------------
# Given / assumed parameters
# ----------------------------

# [beam_number , beam_current , beam_pitch , FOV_area, 
# Expected_n_FOV_tofindmark = [first mark, second mark, third mark], n_realign_per_grid]
settings1100 = [25, 4e-9, 8e-6, 64e-12, [4,1,1], 0.01]
settings2200 = [2791, 0.5e-9, 100e-6, 1e-12, [300,30,1], 1]
settings_test = [0,0,0,0,[0,0,0],0]

beam_number, beam_current, beam_pitch, FOV_area, Expected_n_FOV_tofindmark, n_realign_per_grid = settings2200

# Following variables are the same for 1100 and 2200
optical_accuracy = 10e-6       # m
SNR_inspection = 10                       # signal to noise ratio during inspection
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

N_SE_required = SNR_inspection**2         # number of detected SEs to make image
stage_overhead_time_alignment_per_move = 2*np.sqrt(2/a*(np.sqrt(FOV_area/2))) 
stage_overhead_time_inspection = 2*np.sqrt(2/a*(np.sqrt(FOV_area/2))) * (np.ceil(beam_pitch/np.sqrt(FOV_area))-1)

# Pixel scan time
pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ Collector_efficiency) / (beam_current / e_charge)

# Number of FOV images needed to find mark
FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1)          # s

# Alignment time
initial_alignment_time, time_list = show_time(procedure,Expected_n_FOV_tofindmark)
realignment_time = time_list[-1]

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
pixel_area = pixel_width**2



print(f"FOV scan time = {FOV_scan_time:.3f} s")
print(f"Number of detected SEs to make image = {N_SE_required:.0f}")
print(f"Stage overhead time for alignment = {stage_overhead_time_alignment_per_move*np.sum(Expected_n_FOV_tofindmark):.3f} s")
print(f"Stage overhead time for inspection = {stage_overhead_time_inspection:.3f} s")
print(f"Number of FOV images to find three marks = {np.sum(Expected_n_FOV_tofindmark):.0f}")
print(f"Initial alignment time = {initial_alignment_time:.8f} s")
print(f"Beam scan rate = {beam_scan_rate*1e12:.0f} µm²/s")
print(f"Scan rate = {scan_rate*1e6*3600:.1f} mm²/h")
print(f"Grid area = {grid_area*1e6:.5f} mm²")
print(f"Grid scan time = {grid_scan_time:.5f} s")
print(f"Pixel scan time = {pixel_scan_time*1e9:.3f} ns")
print(f"Pixel area = {pixel_area*1e18:.3f} nm²")
print(f"Number of pixels = {pixels:.0f}")

# Analysis of throughput losses

scan_time_per_alignement = grid_scan_time / n_realign_per_grid
scanning_fraction = scan_time_per_alignement / (realignment_time + scan_time_per_alignement)
effective_throughput = scan_rate * scanning_fraction
absolute_throughput_loss = scan_rate - effective_throughput
relative_troughpit_loss = absolute_throughput_loss/scan_rate
print("#################################################################")
print(f"Scan time per alignment = {scan_time_per_alignement:.3f} s")
print(f"Realignment time = {realignment_time:.3f} s")
print(f"Scanning fraction = {scanning_fraction:.6f}")
print(f"Effective throughput = {effective_throughput*1e6*3600} mm²/h")
print(f"Absolute throughput loss = {absolute_throughput_loss*1e6*3600} mm²/h")
print(f"Relative throughput loss = {relative_troughpit_loss*100} %")