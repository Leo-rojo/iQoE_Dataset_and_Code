import pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
colori=cm.get_cmap('tab10').colors
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 50}
plt.rc('font', **font_general)

# create a figure for the data
figData = pylab.figure()
ax = pylab.gca()
stile = ['-', ':','--','-.']
a=np.arange(0,8,1)
barWidth=3
b=[i+barWidth-0.05 for i in a]
c=[i+barWidth-0.05 for i in b]
d=[i+barWidth-0.05 for i in c]#colori[0],colori[1],colori[2],colori[4],'r'
plt.plot(a, [1,2,3,4,5,6,7,8],stile[0],color ='r',linewidth=7,label='iQoE')
plt.plot(b, [1,2,3,4,5,6,7,8],stile[2], color =colori[0],linewidth=7,label='RS+XSVR')
plt.plot(b, [1,2,3,4,5,6,7,8],stile[2], color =colori[2],linewidth=7,label='GS+XSVR')
plt.plot(c, [1,2,3,4,5,6,7,8],stile[1], color =colori[1],linewidth=7,label='UC+XSVR')
plt.plot(d, [1,2,3,4,5,6,7,8],stile[3], color =colori[4],linewidth=7,label='QBC+XSVR')
plt.plot(d, [1,2,3,4,5,6,7,8],stile[3], color =colori[5],linewidth=7,label='IGS+XSVR')
# leg = ['CU+SVR', 'GS+SVR', 'QBC+SVR', 'iGS+SVR'] + ['RS+SVR', 'iQoE']  # iQoE=igs+SVR
# colors = [colori[1], colori[2], colori[4], colori[5]] + [colori[0], 'r']
# stile = [':', '--', '-.', '-.'] + ['--', '-']
# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=6,frameon=False)
figLegend.savefig("legendss_combined.pdf",bbox_inches='tight')
plt.close()
plt.close()