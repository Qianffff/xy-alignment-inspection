import numpy as np
import matplotlib.pyplot as plt
from eScan_numbers import N_SE_required, SE_escape_factor, SE_yield, collector_efficiency, e, pixel_width, beam_overhead_rate, get_time, settings2200

def relative_throughput(settings,align_method):
    beam_number = settings[1]
    beam_current = settings[2]
    beam_pitch = settings[3]
    FOV_area = settings[4]
    
    if align_method == "area":
        scanned_area_per_alignment = 24.17*1e-6
    elif align_method == "grid":
        grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number # sqrt(3/4) due to hexagonal grid shape
        scanned_area_per_alignment = grid_area
    elif align_method == "time":
        scanned_area_per_alignment = 24.17e-6
    local_alignment_time = get_time(settings,'local')[0]
    
    pixels = int(np.sqrt(FOV_area)/pixel_width)
    
    if beam_number==2791.0:
        if beam_current == 5e-10:
            print(local_alignment_time)
            
    pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ collector_efficiency) / (beam_current/e)
    FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1) # s
    beam_scan_rate = FOV_area / FOV_scan_time
    scan_rate = beam_number * beam_scan_rate
    scan_time_per_alignment = scanned_area_per_alignment / scan_rate
    scanning_fraction = scan_time_per_alignment / (local_alignment_time + scan_time_per_alignment)
    relative_throughput_loss = 1 - scanning_fraction
    return scan_rate, relative_throughput_loss

def i(method,align_method,n=1000):
    import copy
    settings = copy.deepcopy(settings2200)
    x = np.zeros(n)
    y = np.zeros(n)
    if method == "number":
        beam_number = np.geomspace(20,250000000,n)
        beam_current = np.ones(n)*settings[2]
    elif method == "current":
        beam_number = np.ones(n)*settings[1]
        beam_current = np.geomspace(0.003e-9,1e-3,n)
    elif method == "split":
        distribution = .75 # 1 is full on number
        factor = np.geomspace(0.8e-1,400,n)**2
        beam_number = settings[1]*factor**distribution
        beam_current = settings[2]*factor**(1-distribution)
    for k in range(n):
        settings[1] = beam_number[k]
        settings[2] = beam_current[k]
        x[k], y[k] = relative_throughput(settings,align_method)
    x = x*1e6*3600
    y = y*100
    return x,y


for align_method in ["area","grid","time"]:
    plt.figure()
    for method in ["number","current","split"]:
        x_arr, y_arr = i(method,align_method)
        plt.plot(x_arr, y_arr,label=method)
    plt.plot([88.128,1251.273],[0.018,1.015],'k.',markersize=5)
    plt.axvline(17978.985,-0.5,15,color='grey', linestyle='--',label='expected throughput eScan 3000')
    plt.title(align_method)
    plt.legend()
    plt.xlim(1200,25000)
    plt.ylim(-0.5,15)
    plt.xscale("log")
    plt.show(block= align_method=='time')

