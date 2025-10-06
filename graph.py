import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def g(x,a,b,c,d):
    return np.arctan((x-a)*b)*d+c

x = [88.128,1251.273,17978.985,100816.737,403266.946,16654552.21]
y = [0.018,1.015,10.731,40.267,72.947,99.118]

x_arr = np.logspace(np.log10(np.min(x)/5),np.log10(np.max(x)*2),100)
popt,pcov = curve_fit(g,x,y,maxfev=10000)
y_arr = g(x_arr,popt[0],popt[1],popt[2],popt[3])

plt.figure(figsize=(6,6))
plt.plot(x[3:],y[3:],'k.')
labels = ['eScan1100','eScan2200','eScan3000']
plt.grid()
plt.plot(x_arr,y_arr,zorder=0)
plt.scatter(x[0],y[0], c='k', marker='.', s=200)
plt.scatter(x[1],y[1], c='k', marker='.', s=200)
plt.scatter(x[2],y[2], c='k', marker='.', s=200)
plt.text(x[0],y[0]+2,labels[0],ha='center')
plt.text(x[1],y[1]+2,labels[1],ha='center')
plt.text(x[2],y[2]+2,labels[2],ha='center')
plt.xlim([min(x_arr),max(x_arr)])
plt.xscale('log')
plt.xlabel('Scan rate (mm²/h)')
plt.ylabel('Relative througput loss (%)')
plt.title('Relative time spent on alignment vs. ideal throughput')
plt.tight_layout()
plt.show(block = True)