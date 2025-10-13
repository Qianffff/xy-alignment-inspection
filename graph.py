import numpy as np
import matplotlib.pyplot as plt
from eScan_numbers import N_SE_required, SE_escape_factor, SE_yield, collector_efficiency, e, pixel_width, beam_overhead_rate, get_time, settings2200

def relative_throughput(settings,align_method):
    if align_method == "area":
        scanned_area_per_alignment = 24.17e-6
    elif align_method == "grid":
        machine = settings[0]
        beam_pitch = settings[3]
        beam_number = settings[1]
        if machine == '1100': grid_area = (beam_pitch**2) * beam_number
        else: grid_area = (np.sqrt(3/4)*beam_pitch**2) * beam_number # sqrt(3/4) due to hexagonal grid shape
        scanned_area_per_alignment = grid_area
    elif align_method == "time":
        scanned_area_per_alignment = 24.17e-6
    local_alignment_time = get_time(settings[7],'local')[0]
    FOV_area = settings[4]
    pixels = int(np.sqrt(FOV_area)/pixel_width)
    beam_current = settings[2]
    pixel_scan_time = ((N_SE_required / SE_escape_factor) / SE_yield/ collector_efficiency) / (beam_current/e)
    FOV_scan_time = pixel_scan_time * pixels**2 + beam_overhead_rate*pixels*pixel_width*(pixels-1) # s
    beam_scan_rate = FOV_area / FOV_scan_time
    beam_number = settings[1]
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
        factor = np.geomspace(0.8e-1,400,n)
        beam_number = settings[1]*factor
        beam_current = settings[2]*factor
    for k in range(n):
        settings[1] = beam_number[k]
        settings[2] = beam_current[k]
        x[k], y[k] = relative_throughput(settings,align_method)
    x = x*1e6*3600
    y = y*100
    print(np.min(x))
    return x,y # REMOVE beam_overhead_rate from get_time function at the end to see what happens!!!


for align_method in ["area","grid","time"]:
    plt.figure()
    for method in ["number","current","split"]:
        x_arr, y_arr = i(method,align_method)
        plt.plot(x_arr, y_arr,label=method)
    plt.title(align_method)
    plt.legend()
    plt.xlim(10,1e10)
    plt.ylim(-0.5,105)
    plt.xscale("log")
    plt.show(block= align_method=='time')

