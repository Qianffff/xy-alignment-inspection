import numpy as np

# Calculate the time of an alignment procedure
def get_time(procedure,type='local'):
    total_time = 0
    time_breakdown = {}
    i = 1
    for mark in procedure:
        # Include stage overhead time (for movements to, from, and between marks)
        if type == 'local': total_time += stage_overhead_time_local_alignment
        if type == 'global': total_time += stage_overhead_time_mark_to_mark
        
        time_breakdown['Mark ' + str(i)] = {}
        j = 1
        for step in mark:
            time_breakdown['Mark ' + str(i)]['Step ' + str(j)] = {}
            n, pixel_width, SNR = step # Get the parameters that define the step
            
            # Calculate intermediate parameters used to find the time
            scan_time_per_pixel = SNR**2/(SE_yield*SE_escape_factor*collector_efficiency * (beam_current/e))
            pixels = FOV_width / pixel_width
            
            beam_overhead_time = beam_overhead_rate*(pixels*pixel_width)*(pixels-1) # Beam overhead time for one FOV
            FOV_time = scan_time_per_pixel*pixels**2 + beam_overhead_time # Time of one FOV
            step_time = FOV_time * n + stage_overhead_time_per_FOV * n + latency # Time of the step
            
            total_time += step_time # Update total time of the procedure

            time_breakdown['Mark ' + str(i)]['Step ' + str(j)]['Total FOV scan time'] = scan_time_per_pixel*pixels**2 * n
            time_breakdown['Mark ' + str(i)]['Step ' + str(j)]['Total FOV beam overhead time'] = beam_overhead_rate*(pixels*pixel_width)*(pixels-1) * n
            time_breakdown['Mark ' + str(i)]['Step ' + str(j)]['Total stage overhead time'] = stage_overhead_time_per_FOV * n
            time_breakdown['Mark ' + str(i)]['Step ' + str(j)]['Latency'] = latency

            j += 1
        if type == 'local': time_breakdown['Mark ' + str(i)]['Mark stage movement time'] = stage_overhead_time_local_alignment
        if type == 'global': time_breakdown['Mark ' + str(i)]['Mark stage movement time'] = stage_overhead_time_mark_to_mark
        i += 1
    return total_time, time_breakdown


def print_alignment_data(data, procedure, indent=0):
    grand_total = 0  # total across all marks
    mark_totals = {}

    # First pass to compute all totals
    for mark, contents in data.items():
        mark_total = 0
        for key, val in contents.items():
            if key.startswith("Step"):
                mark_total += sum(val.values())
            elif key == "Mark stage movement time":
                mark_total += val
        mark_totals[mark] = mark_total
        grand_total += mark_total

    # Print grand total at top
    print(f"Total alignment time: {grand_total:.6f} s\n")

    # Second pass to print formatted output
    i = 0
    for mark, contents in data.items():
        print("   " * indent + f"{mark}: {mark_totals[mark]:.6f} s")
        j = 0
        for key, val in contents.items():
            if key.startswith("Step"):
                step_total = sum(val.values())
                print("   " * (indent + 2) + f"{key} (N_FOV = {procedure[i][j][0]}, Pixel Width = {procedure[i][j][1]}, SNR = {procedure[i][j][2]}): {step_total:.6f} s")
                for subkey, time in val.items():
                    print("   " * (indent + 4) + f"{subkey}: {time*1e3:.2f} ms")
            elif key == "Mark stage movement time":
                print("   " * (indent + 2) + f"{key}: {val:.6f} s")
            j += 1
        i += 1



###################### Machine-speficic parameters #############################

beam_number_1100 = 25
beam_current_1100 = 4e-9 # A
beam_pitch_1100 = 8e-6 # m
FOV_area_1100 = 64e-12 # m²
n_align_per_grid_1100 = 6.62e-5 # Based on one local alignment per 24 mm²

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
cross_length = 4*1e-6 # m
pixel_width = 5e-9 # m
beam_overhead_rate = 0.1 # s/m
latency = (0.1 + 1 + 0.1)*1e-3 # s

stage_speed = 0.4 # m/s
a = 2*9.81 # stage acceleration (m/s²)
stage_settling_time = 1*1e-3 # s
mark_distance_global = 0.15 # m
total_stage_movement_local_alignment = 42e-3 # (in m) 42e-3 is the distance to go from the center of a 26x33 mm die to the corner and back again

# Constants
e = 1.602e-19 # C


#################### Define alignment procedures ###########################


# An alignment procedure is built up like this:
#
# procedure = [mark_1,mark_2,mark_3,...] (5 marks for global alignment, 1 mark for local alignment)
# mark_i = [step_i_1,step_i_2,...]
# step_i.j = [n,pixel_width,SNR]

# Create global alignment procedure for the 2200:

n_min = (2*cross_length)**2 / FOV_area_2200 # Minimum number of FOVs needed to image the full cross (with some margin)

step_1_1 = [300, 20e-9, 5]
step_1_2 = [n_min, 5e-9, 10]
mark_1 = [step_1_1,step_1_2]

step_2_1 = [30, 20e-9, 5]
step_2_2 = [n_min, 5e-9, 10]
mark_2 = [step_2_1, step_2_2]

step_3_1 = [n_min, 5e-9, 10]
mark_3 = [step_3_1]

step_4_1 = [n_min, 5e-9, 10]
mark_4 = [step_4_1]

step_5_1 = [n_min, 5e-9, 10]
mark_5 = [step_5_1]

procedure_2200_global = [mark_1,mark_2,mark_3,mark_4,mark_5]

# Create local alignment procedure for the 2200:

step_1_1 = [n_min, 5e-9, 10]
mark_1 = [step_1_1]
procedure_2200_local = [mark_1]

# Create global alignment procedure for the 1100:

n_min = (2*cross_length)**2 / FOV_area_1100 # Minimum number of FOVs needed to image the full cross (with some margin)

step_1_1 = [5, 20e-9, 5]
step_1_2 = [n_min, 5e-9, 10]
mark_1 = [step_1_1,step_1_2]

step_2_1 = [n_min, 5e-9, 10]
mark_2 = [step_2_1]

step_3_1 = [n_min, 5e-9, 10]
mark_3 = [step_3_1]

step_4_1 = [n_min, 5e-9, 10]
mark_4 = [step_4_1]

step_5_1 = [n_min, 5e-9, 10]
mark_5 = [step_5_1]

procedure_1100_global = [mark_1,mark_2,mark_3,mark_4,mark_5]

# Create local alignment procedure for the 1100:

step_1_1 = [n_min, 5e-9, 10]
mark_1 = [step_1_1]
procedure_1100_local = [mark_1]

################## Quick switch between machine settings #########################


# [beam_number , beam_current , beam_pitch , FOV_area, n_realign_per_grid]
settings1100 = ['1100', beam_number_1100, beam_current_1100, beam_pitch_1100, FOV_area_1100, n_align_per_grid_1100, procedure_1100_global, procedure_1100_local]
settings2200 = ['2200', beam_number_2200, beam_current_2200, beam_pitch_2200, FOV_area_2200, n_align_per_grid_2200, procedure_2200_global, procedure_2200_local]
settings3000 = ['3000', beam_number_2200*10.7, beam_current_2200*1.344, beam_pitch_2200, FOV_area_2200, n_align_per_grid_2200, procedure_2200_global, procedure_2200_local]
settings_test = ['test', beam_number_2200*10.7, beam_current_2200*1.344, beam_pitch_2200, FOV_area_2200, n_align_per_grid_2200, procedure_2200_global, procedure_2200_local]

machine, beam_number, beam_current, beam_pitch, FOV_area, n_realign_per_grid, procedure_global, procedure_local = settings3000

############################# Calculations ####################################

FOV_width = np.sqrt(FOV_area)
pixels = int(FOV_width/pixel_width)

N_SE_required = SNR_inspection**2 # number of detected SEs to make image

stage_overhead_time_per_FOV = 2*np.sqrt(2*(FOV_width/2)/a) + stage_settling_time
stage_overhead_time_per_grid = stage_overhead_time_per_FOV * (np.ceil(beam_pitch/FOV_width)-1)
stage_overhead_time_mark_to_mark = mark_distance_global / stage_speed
stage_overhead_time_local_alignment = total_stage_movement_local_alignment / stage_speed

# Pixel scan time
pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ collector_efficiency) / (beam_current/e)

# Number of FOV images needed to find mark
FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1) # s

# Alignment time

global_alignment_time, time_breakdown_global_alignment = get_time(procedure_global,'global')
# print("##################################################################")
# print_alignment_data(time_breakdown_global_alignment,procedure_global)
# print("##################################################################")

local_alignment_time,time_breakdown_local_alignment = get_time(procedure_local,'local')
print("##################################################################")
print_alignment_data(time_breakdown_local_alignment,procedure_local)
print("##################################################################")

# Beam scan rate (per beam)
beam_scan_rate = FOV_area / FOV_scan_time

# Total scan rate (all beams)
scan_rate = beam_number * beam_scan_rate

# Grid area
if machine == '1100': grid_area = (beam_pitch**2) * beam_number
else: grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number # sqrt(3/4) due to hexagonal grid shape

# Grid scan time
grid_scan_time = grid_area / scan_rate + stage_overhead_time_per_grid

# Pixel area (approx)
pixels_per_FOV = FOV_scan_time / pixel_scan_time
pixel_area = pixel_width**2

n_eFOVs_to_align_global = 0
for mark in procedure_global:
    for step in mark:
        n_eFOVs_to_align_global += step[0]


# Assume that you need realignment at fixed area intervals:
scanned_area_per_alignment = 24.17*1e-6 # m² # This means local alignment must be done for every 24.17 mm² that is scanned. (24.17 mm² is the grid area of the eScan2200.)
scan_time_per_alignment = scanned_area_per_alignment / scan_rate
scanning_fraction = scan_time_per_alignment / (local_alignment_time + scan_time_per_alignment)
effective_throughput = scan_rate * scanning_fraction
absolute_throughput_loss = scan_rate - effective_throughput
relative_troughput_loss = absolute_throughput_loss/scan_rate # = 1 - scanning_fraction

# # Older version: assume you need realignment 'n_realign_per_grid' times per grid (independent of the grid area)
# scan_time_per_alignement = grid_scan_time / n_realign_per_grid
# scanning_fraction = scan_time_per_alignement / (local_alignment_time + scan_time_per_alignement)
# effective_throughput = scan_rate * scanning_fraction
# absolute_throughput_loss = scan_rate - effective_throughput
# relative_troughput_loss = absolute_throughput_loss/scan_rate


###################### Print numbers #########################

print(f"Pixel area = {pixel_area*1e18:.0f} nm²")
print(f"Pixel scan time = {pixel_scan_time*1e9:.2f} ns")

print(f"Number of pixels in FOV = {pixels**2:.0f}")
print(f"FOV scan time = {FOV_scan_time:.4f} s")

print(f"Grid area = {grid_area*1e6:.5f} mm²")
print(f"Grid scan time = {grid_scan_time:.5f} s")

print(f"Number of FOV images for global alignment = {n_eFOVs_to_align_global:.0f}")
print(f"Global alignment time = {global_alignment_time:.4f} s")

print(f"SNR during inspection = {SNR_inspection:.0f}")
print(f"Stage overhead time during global alignment = {stage_overhead_time_per_FOV*n_eFOVs_to_align_global:.6f} s")
print(f"Stage overhead time per inspected grid = {stage_overhead_time_per_grid:.6f} s")
print(f"Mark-to-mark stage overhead time (for global align) = {stage_overhead_time_mark_to_mark*1e3:.3f} ms")

print(f"Beam scan rate = {beam_scan_rate*1e12:.0f} µm²/s")
print(f"Scan rate = {scan_rate*1e6*3600:.3f} mm²/h")

# Analysis of throughput losses
print("#################################################################")

print(f"Scan time per alignment = {scan_time_per_alignment:.3f} s")
print(f"Local alignment time = {local_alignment_time:.5f} s")
print(f"Scanning fraction = {scanning_fraction:.5f}")
print(f"Effective throughput = {effective_throughput*1e6*3600:.2f} mm²/h")
print(f"Absolute throughput loss = {absolute_throughput_loss*1e6*3600:.2f} mm²/h")
print(f"Relative throughput loss = {relative_troughput_loss*100:.3f} %")