import pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D


colori=cm.get_cmap('tab10').colors
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 50}
plt.rc('font', **font_general)


# create a figure for the data
#1-20, 21-40, 41-60, 61-80, and81-100
figData = pylab.figure()
stile = ['-', ':','--','-.']
a=np.arange(0,8,1)
barWidth=3
b=[i+barWidth-0.05 for i in a]
c=[i+barWidth-0.05 for i in b]
d=[i+barWidth-0.05 for i in c]#colori[0],colori[1],colori[2],colori[4],'r'
plt.plot(a, [1,2,3,4,5,6,7,8],marker='o', color='w', label='Scores 1-20 (bad)',markerfacecolor='blue', markersize=22,markeredgecolor='black')
plt.plot(b, [1,2,3,4,5,6,7,8], marker='v', color='w', label='Scores 21-40 (poor)',markerfacecolor='cyan', markersize=22,markeredgecolor='black')
plt.plot(b, [1,2,3,4,5,6,7,8], marker='d', color='w', label='Scores 41-60 (fair)',markerfacecolor='green', markersize=22,markeredgecolor='black')
plt.plot(c, [1,2,3,4,5,6,7,8], marker='p', color='w', label='Scores 61-80 (good)',markerfacecolor='yellow', markersize=22,markeredgecolor='black')
plt.plot(d, [1,2,3,4,5,6,7,8], marker='s', color='w', label='Scores 81-100 (excellent)',markerfacecolor='red', markersize=22,markeredgecolor='black')
#plt.plot(d, [1,2,3,4,5,6,7,8] ,marker='s', color='w', label='homogenous scores',markerfacecolor='none', markersize=30,markeredgecolor='blue', markeredgewidth=3)
ax = pylab.gca()

# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=5,frameon=False, handletextpad=-0.2, columnspacing=0.5)
figLegend.savefig("legendss_sampler.pdf",bbox_inches='tight')
plt.close()
plt.close()