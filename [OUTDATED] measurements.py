with open('measurements.txt','r') as file:
    measurements = file.read().splitlines()
HMIeScan1100_2000_absolute_distance_errors = [float(line.strip()) for line in measurements]

import matplotlib.pyplot as plt
#plotting of histogram
plt.figure(figsize=(8,6))
plt.hist(HMIeScan1100_2000_absolute_distance_errors, bins=20, color='skyblue', edgecolor='k')
plt.xlabel("Error (nm)")
plt.ylabel("Counts")
plt.title('Histogram: error of alignment detection algorithm (HMI eScan 1100 settings)')
plt.show(block = True)

import numpy as np
print('std = ', np.std(HMIeScan1100_2000_absolute_distance_errors))
print('mean = ', np.mean(HMIeScan1100_2000_absolute_distance_errors))