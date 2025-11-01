import numpy as np
import matplotlib.pyplot as plt
from eScan_numbers import N_SE_required, SE_escape_factor, SE_yield, collector_efficiency, e, pixel_width, beam_overhead_rate, get_time, settings2200, total_time, global_alignment_time, grid_scan_time,beam_pitch,stage_speed
import copy

def relative_throughput(settings,align_method,method):
    beam_number = settings[1]
    beam_current = settings[2]
    FOV_area = settings[4]
    
    pixels = int(np.sqrt(FOV_area)/pixel_width)
    pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ collector_efficiency) / (beam_current/e)
    FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1) # s
    beam_scan_rate = FOV_area / FOV_scan_time
    scan_rate = beam_number * beam_scan_rate
    
    if align_method == "area":
        scanned_area_per_alignment = 24.17*1e-6
    elif align_method == "grid":
        grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number
        scanned_area_per_alignment = grid_area
    elif align_method == "time":
        accuracy = 16.36*1e-9 # (m)
        time_error_rate = 0.1786*1e-9 # (m_error/s)
        stage_movement_error_rate = 0.1*1e-6 # (m_error/m_stage_movement)
        c = 0.9e-10 # Proportionality constant in model of wafer expansion due to the beams heating it up

        t1 = 0.0325 # Stage movement time from alignment mark to inspection area (s)
        d1 = 13*1e-3 # Stage movement distance fram alignment mark to inspection area (m)
        e1 = time_error_rate*t1 + stage_movement_error_rate*d1 # Calculate drift buildup

        v = 0.124843*1e-3 # Stage speed during inspection
        d = (accuracy-e1)/(stage_movement_error_rate + time_error_rate/v + beam_number*c) # Travel distance of the stage during inspection before realignment is necessary
        scanned_area_per_alignment = d*np.sqrt(FOV_area)*beam_number
    
    local_alignment_time = get_time(settings,'local')[0]
    global_alignment_time = get_time(settings,'global')[0]
    
    if method == 'Existing method':
        local_alignment_time = local_alignment_time
        global_alignment_time = global_alignment_time
    elif method == 'Dedicated e-beam':
        global_alignment_time = 6.9306335
        local_alignment_time = 0
    elif method == 'Diffraction':
        global_alignment_time = 4.116528
        local_alignment_time = 0
    elif method == 'Computational':
        global_alignment_time = 0
        local_alignment_time = 0
    elif method == 'Multibeam':
        global_alignment_time = global_alignment_time
        local_alignment_time = 0.1142
    
    scan_time_per_alignment = scanned_area_per_alignment / scan_rate
    scanning_fraction = ((total_time - global_alignment_time)/total_time) * scan_time_per_alignment / (local_alignment_time + scan_time_per_alignment)
    relative_throughput_loss = 1 - scanning_fraction
    return scan_rate, relative_throughput_loss

def i(align_method,method,n=1000):
    settings = copy.deepcopy(settings2200)
    x = np.zeros(n)
    y = np.zeros(n)
    beam_number = np.geomspace(20,250000000,n)
    beam_current = np.ones(n)*settings[2]
    for k in range(n):
        settings[1] = beam_number[k]
        settings[2] = beam_current[k]
        x[k], y[k] = relative_throughput(settings,align_method,method)
    x = x*1e6*3600
    y = y*100
    return x,y

plt.figure()
colors = ['red','orange','green']
for j, align_method in enumerate(["area","time","grid"]):
    x_arr, y_arr = i(align_method,method='Existing method')
    if align_method == 'area': label = 'Realign every 24 mm²'
    elif align_method == 'grid': label = 'Realign every grid image'
    elif align_method == 'time': label = 'Realign if drift exceeds 16 nm'
    plt.plot(x_arr, y_arr,label=label,color=colors[j])
    plt.axvline(1246.71,-0.5,15,color='grey', linestyle='--')
    plt.axvline(17978.985,-0.5,15,color='grey', linestyle='--')

# plt.title('Alignment time vs. throughput (do-nothing solution)')

# For report
plt.text(1100,-0.9,'eScan 2200')
plt.text(14700,-0.9,'eScan 3000')
plt.xlabel('Throughput (mm²/h)')
plt.ylabel('Alignment time (%)')
plt.yticks(np.round(np.arange(0, 18.1, 3),0))
plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')

# For final presentation
# plt.text(1100,-1.4,'eScan 2200',fontsize=15)
# plt.text(13900,-1.4,'eScan 3000',fontsize=15)
# plt.xlabel('Throughput (mm²/h)',fontsize=15)
# plt.ylabel('Alignment time (%)',fontsize=15)
# plt.yticks(np.round(np.arange(0, 18.1, 3),0),fontsize=15)
# plt.xticks(fontsize=15)
# plt.legend(fontsize=14,bbox_to_anchor=(0.05, 1), loc='upper left')

plt.xlim(1100,25000)
plt.ylim(0,18)
plt.xscale("log")
plt.tight_layout()
plt.show(block = False)

plt.figure()
colors = ['orange','purple','lightblue','blue','green']
for j, method in enumerate(['Existing method','Multibeam','Dedicated e-beam','Diffraction','Computational']):
    x_arr, y_arr = i('time', method)
    plt.plot(x_arr, y_arr,label=method,color=colors[j])

plt.axvline(1246.71,-0.5,15,color='grey', linestyle='--')
plt.axvline(17978.985,-0.5,15,color='grey', linestyle='--')

# plt.title('Alignment time vs. throughput (realignment based on drift build-up)')

# For report
plt.xlabel('Throughput (mm²/h)')
plt.ylabel('Alignment time (%)')
plt.yticks(np.round(np.arange(0, 5, 1),0))
plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
plt.text(1100,-0.48,'eScan 2200',fontsize=10)
plt.text(14700,-0.48,'eScan 3000',fontsize=10)

# For final presentation
# plt.xlabel('Throughput (mm²/h)',fontsize=15)
# plt.ylabel('Alignment time (%)',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(np.round(np.arange(0, 5, 1),0),fontsize=15)
# plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left',fontsize=14)
# plt.text(1100,-0.55,'eScan 2200',fontsize=15)
# plt.text(13900,-0.55,'eScan 3000',fontsize=15)

plt.xlim(1100,25000)
plt.ylim(-0.2,4.2)
plt.xscale("log")
plt.tight_layout()
plt.show(block = True)