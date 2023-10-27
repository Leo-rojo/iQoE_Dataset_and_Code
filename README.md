# iQoE Dataset and Code
The repository collects and openly releases the dataset and accompanying code for the results published in the following article: 
    
    Leonardo Peroni, Sergey Gorinsky, Farzad Tashtarian, and Christian Timmerer. 2023. Empowerment of Atypical Viewers via Low-Effort Personalized Modeling of Video Streaming Quality. Proc. ACM Netw. 1, CoNEXT3, Article 17 (December 2023). https://doi.org/10.1145/3629139 

To reproduce the results, please follow the instructions provided in the README.md files located inside the subfolders.

## Requirements

We tested the code with `Python 3.7` with the following additional requirements that can be found in the `requirements.txt` file:

```
Flask==2.2.3
Flask_Cors==3.0.10
Flask_HTTPAuth==4.7.0
k_means_constrained==0.7.0
matplotlib==3.5.3
modAL==0.4.1
modAL==0.55.3917
numpy==1.21.6
pandas==1.3.5
Pillow==9.2.0
scikit_learn==1.0.2
scipy==1.7.3
seaborn==0.13.0
selenium==4.14.0
statsmodels==0.14.0
Werkzeug==2.2.3
xgboost==1.6.2
itu-p1203==1.8.3
```
### L model
For running baseline L model it needs to be installed following https://github.com/TranHuyen1191/BiQPS. 

### FFMPEG
For running experiments that involve creation of videos and calculation of relative IFs like PSNR, SSIM, VMAF FFMPEG need to be installed: https://ffmpeg.org/download.html we used the builded version in https://www.gyan.dev/ffmpeg/builds/. 

### QP values
For calculation of QP values we used ffmpeg_debug_qp (https://github.com/slhck/ffmpeg-debug-qp).