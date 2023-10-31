import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

itera=[str(i)+'_' for i in [6,9,10,11,12,13,1,2,3,4,5,7,8]]

for it in itera:
    nparray=np.load(it+'.npy')
    ## convert your array into a dataframe
    df = pd.DataFrame (nparray)
    df.columns = ["size", "width", "height", "bitrate",'psnr','ssim','vmaf']
    ## save to xlsx file

    filepath = it+'.xlsx'

    df.to_excel(filepath, index=True)

    vmaf_column=nparray[:,6].astype(np.float).tolist()
    ssim_column=nparray[:,5].astype(np.float).tolist()
    psnr_column=nparray[:,4].astype(np.float).tolist()

    plt.plot(vmaf_column,label=it)
plt.legend()



