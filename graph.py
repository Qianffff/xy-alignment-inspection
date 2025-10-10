import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def g(x,a,b,c,d):
    return np.arctan((x-a)*b)*d+c

x = [88.128,1251.273,17978.985,100816.737,403266.946,16654552.21]
y = [0.018,1.015,10.731,40.267,72.947,99.118]

x_arr = np.logspace(np.log10(np.min(x)/5),np.log10(x[2]*2),100)
popt,pcov = curve_fit(g,x,y,maxfev=10000)
y_arr = g(x_arr,popt[0],popt[1],popt[2],popt[3])

plt.figure(figsize=(6,6))
plt.plot(x[3:],y[3:],'k.')
#plt.plot(x_arr,y_arr,zorder=0)

xmin = min(x_arr)
xmax = max(x_arr)
ymin = -0.5
ymax = 12
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])

p1x,p1y = [x[0],g(x[0],popt[0],popt[1],popt[2],popt[3])]
p2x,p2y = [x[1],g(x[1],popt[0],popt[1],popt[2],popt[3])]
p3x,p3y = [x[2],g(x[2],popt[0],popt[1],popt[2],popt[3])]

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



# # Bar diagram

# # Data
# machines = ["eScan1100", "eScan2200", "eScan3000"]
# values = [0.12, 1.75, 25]

# # Create bar chart
# plt.figure(figsize=(8, 6))
# bars = plt.bar(machines, values, color="deepskyblue")

# # Add labels on bars
# for bar, value in zip(bars, values):
#     plt.text(bar.get_x() + bar.get_width()/2, 
#              bar.get_height() - (0.1 * bar.get_height()),  # slightly below top
#              f"{value}", 
#              ha="center", va="bottom", fontsize=12, color="black")

# # Customize title, labels, and ticks
# plt.title("Percentage of Wafer Scanned per Hour", fontsize=14)
# plt.ylabel("Percentage", fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# # Set y-axis limit (similar to your chart)
# plt.ylim(0, 30)

# plt.show()