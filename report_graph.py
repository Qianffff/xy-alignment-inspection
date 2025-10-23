import numpy as np
import matplotlib.pyplot as plt
from eScan_numbers import N_SE_required, SE_escape_factor, SE_yield, collector_efficiency, e, pixel_width, beam_overhead_rate, get_time, settings2200, total_time, global_alignment_time, total_stage_movement_local_alignment
import copy

def relative_throughput(settings,align_method):
    beam_number = settings[1]
    beam_current = settings[2]
    beam_pitch = settings[3]
    FOV_area = settings[4]
    
    pixels = int(np.sqrt(FOV_area)/pixel_width)
    pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ collector_efficiency) / (beam_current/e)
    FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1) # s
    beam_scan_rate = FOV_area / FOV_scan_time
    scan_rate = beam_number * beam_scan_rate
    
    if align_method == "area":
        scanned_area_per_alignment = 24.17*1e-6
    elif align_method == "grid":
        grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number # sqrt(3/4) due to hexagonal grid shape
        scanned_area_per_alignment = grid_area
    elif align_method == "time":
        
        accuracy = 10*np.sqrt(2)*1e-9
        time_error_rate = 0.05e-9 # m_error/s
        stage_movement_error_rate = 0.1e-6 # m_error/m_stage_movement
        
        total_movement_per_grid = (np.sqrt(3/4)*beam_pitch**2)/FOV_area * np.sqrt(FOV_area) + beam_pitch*np.sqrt(beam_number) + total_stage_movement_local_alignment
        grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number # sqrt(3/4) due to hexagonal grid shape
        scanned_area_per_alignment = (grid_area / total_movement_per_grid) * (accuracy / stage_movement_error_rate)
        scanned_time_per_alignment = scanned_area_per_alignment/scan_rate
        time_error = time_error_rate * scanned_time_per_alignment
        factor = time_error/accuracy
        scanned_area_per_alignment *= 1 - factor
    
    local_alignment_time = get_time(settings,'local')[0]
    
    scan_time_per_alignment = scanned_area_per_alignment / scan_rate
    scanning_fraction = ((total_time - global_alignment_time)/total_time) *scan_time_per_alignment / (local_alignment_time + scan_time_per_alignment)
    relative_throughput_loss = 1 - scanning_fraction
    return scan_rate, relative_throughput_loss

def i(align_method,n=1000):
    settings = copy.deepcopy(settings2200)
    x = np.zeros(n)
    y = np.zeros(n)
    beam_number = np.geomspace(20,250000000,n)
    beam_current = np.ones(n)*settings[2]
    for k in range(n):
        settings[1] = beam_number[k]
        settings[2] = beam_current[k]
        x[k], y[k] = relative_throughput(settings,align_method)
    x = x*1e6*3600
    y = y*100
    return x,y

plt.figure()
colors = ['red','orange','green']
for j, align_method in enumerate(["area","time","grid"]):
    x_arr, y_arr = i(align_method)
    if align_method == 'area': label = 'Realign every 24 mm²'
    elif align_method == 'grid': label = 'Realign every grid image'
    elif align_method == 'time': label = 'Realign if drift exceeds 10 nm'
    plt.plot(x_arr, y_arr,label=label,color=colors[j])
    plt.axvline(1246.71,-0.5,15,color='grey', linestyle='--')
    plt.axvline(17978.985,-0.5,15,color='grey', linestyle='--')
    plt.text(1100,-0.9,'eScan 2200',fontsize=10)
    plt.text(14700,-0.9,'eScan 3000',fontsize=10)
    plt.xlabel('Throughput (mm²/h)')
    plt.ylabel('Alignment time (%)')
    plt.legend()
    plt.xlim(1100,25000)
    plt.ylim(0,18)
    plt.xscale("log")
    plt.tight_layout()
    plt.show(block= align_method=='grid')