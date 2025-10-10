import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def g(x,a,b,c,d):
    return np.arctan((x-a)*b)*d+c

# Realignment: area based (24 mm²), settings3000 = ['3000', beam_number_2200*10.7, beam_current_2200*1.344, beam_pitch_2200, FOV_area_2200
x1 = [88.128,1251.273,17978.985,100816.737,403266.946,16654552.21]
y1 = [0.018,1.015,10.731,40.267,72.947,99.118]

### Area based
# 
x_area__beam_number = [88.128,1251.273,17994.312]
y_area__beam_number = [0.018,1.015,12.850]

# 
x_area__mixed = [88.128,1251.273,17990.904]
y_area__mixed = [0.018,1.015,6.48]

# 
x_area__beam_current = [88.128,1251.273,17999.792]
y_area__beam_current = [0.018,1.015,4.633]

#
x_area = [x_area__beam_number,x_area__mixed,x_area__beam_current]
y_area = [y_area__beam_number,y_area__mixed,y_area__beam_current]
### Grid based
# 
x_grid__beam_number = [88.128,1251.273,17994.312]
y_grid__beam_number = [0.018,1.015,1.015]

# 
x_grid__mixed = [88.128,1251.273,17990.904]
y_grid__mixed = [0.018,1.015,1.788]

# 
x_grid__beam_current = [88.128,1251.271,17999.792]
y_grid__beam_current = [0.018,1.015,4.632]

### Time / error buildup based
# 
x_time__beam_number = []
y_time__beam_number = []

# 
x_time__mixed = []
y_time__mixed = []

# 
x_time__beam_current = []
y_time__beam_current = []



x_arr = np.logspace(np.log10(np.min(x1)/5),np.log10(x1[2]*2),100)
popt,pcov = curve_fit(g,x1,y1,maxfev=10000)
y_arr = g(x_arr,popt[0],popt[1],popt[2],popt[3])

plt.figure(figsize=(6,6))
plt.plot(x1[3:],y1[3:],'k.')
plt.plot(x_arr,y_arr,zorder=0)

xmin = min(x_arr)
xmax = max(x_arr)
ymin = -0.5
ymax = 12
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])

p1x,p1y = [x1[0],g(x1[0],popt[0],popt[1],popt[2],popt[3])]
p2x,p2y = [x1[1],g(x1[1],popt[0],popt[1],popt[2],popt[3])]
p3x,p3y = [x1[2],g(x1[2],popt[0],popt[1],popt[2],popt[3])]

plt.scatter(p1x,p1y, c='k', marker='.', s=300)
plt.scatter(p2x,p2y, c='k', marker='.', s=300)
plt.scatter(p3x,p3y, c='k', marker='.', s=300)

# labels = ['eScan1100','eScan2200','eScan3000']
# plt.text(x[0],y[0]+2,labels[0],ha='center')
# plt.text(x[1],y[1]+2,labels[1],ha='center')
# plt.text(x[2],y[2]+2,labels[2],ha='center')

# Add projection lines to axes
for px, py in [(p1x,p1y), (p2x,p2y), (p3x,p3y)]:
    # Vertical projection to x-axis
    plt.plot([px, px], [ymin, py], color='lightgray', linestyle='--', linewidth=1, zorder=0)
    # Horizontal projection to y-axis
    plt.plot([xmin, px], [py, py], color='lightgray', linestyle='--', linewidth=1, zorder=0)

plt.xscale('log')
plt.xlabel('Ideal throughput (mm²/h)',fontsize=18)
plt.ylabel('Relative time spent on alignment (%)',fontsize=18)
plt.tick_params(labelsize=18)
#plt.title('Relative time spent on alignment vs. ideal throughput',fontsize=18)
plt.tight_layout()
plt.show(block = True)