import pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
colori=cm.get_cmap('tab10').colors
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 50}
plt.rc('font', **font_general)
#assing a random array of values to the variables
iqoe=[1,2,3,4,5,6,7,8]
rmse_bitrate=[1,2,3,4,5,6,7,8]
rmse_logbitrate=[1,2,3,4,5,6,7,8]
rmse_videoatlas=[1,2,3,4,5,6,7,8]
rmse_ssim=[1,2,3,4,5,6,7,8]
rmse_psnr=[1,2,3,4,5,6,7,8]
rmse_vmaf=[1,2,3,4,5,6,7,8]
rmse_ftw=[1,2,3,4,5,6,7,8]
rmse_sdn=[1,2,3,4,5,6,7,8]
rmse_p1203=[1,2,3,4,5,6,7,8]
rmse_lstm=[1,2,3,4,5,6,7,8]
# create a figure for the data

conta = 0
stile = ['-', '--', '-.', ':', '-', '--', '-','-.', ':','-.', ':']
col=['r',colori[1],colori[2],colori[4],colori[6],colori[7],colori[8],colori[9],'gold','darkblue',colori[5]]
names=['iQoE', 'B', 'G', 'A', 'S', 'R', 'V', 'F', 'N','P', 'L']
ordered_names=['iQoE','F','P','S','A', 'R','L', 'G', 'V', 'N', 'B']
for regr in [iqoe,rmse_bitrate,rmse_logbitrate,rmse_videoatlas,rmse_ssim,rmse_psnr,rmse_vmaf,rmse_ftw,rmse_sdn,rmse_p1203,rmse_lstm]:
    #print(regr)
    plt.plot(regr,stile[names.index(ordered_names[conta])], linewidth='7',color=col[names.index(ordered_names[conta])],label=ordered_names[conta])
    conta += 1
ax = pylab.gca()
# leg = ['CU+SVR', 'GS+SVR', 'QBC+SVR', 'iGS+SVR'] + ['RS+SVR', 'iQoE']  # iQoE=igs+SVR
# colors = [colori[1], colori[2], colori[4], colori[5]] + [colori[0], 'r']
# stile = [':', '--', '-.', '-.'] + ['--', '-']
# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=11,frameon=False,columnspacing=0.6,handletextpad=0.2,handlelength=1.45)
figLegend.savefig("../legendss_mos_dashed.pdf",bbox_inches='tight')
plt.close()
##########################################
figData = pylab.figure()
ax = pylab.gca()
conta = 0
markers= ['o', 's', 'D', 'v', '*', 'h', '^', '8', 'P', '<', 'X']
stile = ['-', '--', '-.', ':', '-', '--', '-','-.', ':','-.', ':']
col=['r',colori[1],colori[2],colori[4],colori[6],colori[7],colori[8],colori[9],'gold','darkblue',colori[5]]
names=['iQoE', 'B', 'G', 'A', 'S', 'R', 'V', 'F', 'N','P', 'L']
ordered_names=['iQoE','F','S','A', 'R','P','L', 'G', 'V', 'N', 'B']
for regr in [iqoe,rmse_bitrate,rmse_logbitrate,rmse_videoatlas,rmse_ssim,rmse_psnr,rmse_vmaf,rmse_ftw,rmse_sdn,rmse_p1203,rmse_lstm]:
    #print(regr)
    plt.plot(regr,stile[names.index(ordered_names[conta])], linewidth='7',color=col[names.index(ordered_names[conta])],label=ordered_names[conta])
    conta += 1
# leg = ['CU+SVR', 'GS+SVR', 'QBC+SVR', 'iGS+SVR'] + ['RS+SVR', 'iQoE']  # iQoE=igs+SVR
# colors = [colori[1], colori[2], colori[4], colori[5]] + [colori[0], 'r']
# stile = [':', '--', '-.', '-.'] + ['--', '-']
# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=11,frameon=False,columnspacing=0.6,handletextpad=0.2,handlelength=1.45)
#figLegend.savefig("legendss_pers_dashed.pdf",bbox_inches='tight')
plt.close()

