# iQoE Dataset and Code
The repository collects and openly releases the dataset and accompanying code for the results published in the following article: 
    
    Leonardo Peroni, Sergey Gorinsky, Farzad Tashtarian, and Christian Timmerer. 2023. Empowerment of Atypical Viewers via Low-Effort Personalized Modeling of Video Streaming Quality. Proc. ACM Netw. 1, CoNEXT3, Article 17 (December 2023). https://doi.org/10.1145/3629139 

To reproduce the results, please follow the instructions provided in the readme.md files located inside the subfolders.

### Structure
We support the reproducibility on two levels characterized by different resource requirements and time commitments. Level A (LA) recreates all figures and tables, mostly by running
our Python code. The reproduction performs limited calculations based on data in intermediate representations, e.g., scores by a rater for a personalized series of experiences. Level B (LB) involves advanced computations, such as model training, and carries out in-depth replication from raw
data, allowing also for new data from independent subjective studies and simulations. While LA is
primarily for validating the results of this paper, LB targets the long-term impact of the artifacts
through their reuse in future research.

### Artifact checklist
- Algorithm: Algorithm 1.
- Programs: (LA) Python 3.7 with supporting libraries; (LB) FFmpeg, ffmpeg-debug-qp, Park, MP4Box, BiQPS, HTML, JavaScript, and CSS.
- Models: SVR, RF, GP, and XGB.
- Datasets: (LA) iQoE and Waterloo-IV; (LB) network traces, video chunks, and experience set.
- Hardware: no particular requirements.
- Output: (LA) all figures and tables of the paper; (LB) Fig. 2-13 and 15-18 and Tables 1 and 2.
- Approximate disk space requirement: (LA) 4 GB; (LB) 24 GB.
- Approximate experiment completion time: (LA) under one hour; (LB) about one week.
- Availability: public on GitHub.

### Artifact access
We structure the GitHub repository in folders dedicated to a figure or table. Each folder contains LA and/or LB
subfolders corresponding to the two levels of result reproducibility. The subfolders hold all the data
and code required for generating the respective results.
### Hardware dependencies
There are no specific hardware requirements except for support of
Python. The machine used in our experiments is an Intel i7 with six cores, 2.6-GHz CPUs, 16-GB
RAM, and Windows 10.
### Software dependencies 
The requirements.txt file in the repository lists the software dependencies, with readme.md files in individual subfolders supplying any further details and clarifications.
### Models
SVR, RF, GP, and XGB are from Python’s scikit-learn and xgboost libraries.
### Datasets
In addition to the experiment-specific data in each subfolder, the repository includes a standalone Datasets folder with our iQoE dataset in its three versions used by different
experiments. These versions appear in the .xlsx format in subfolders dataset_34, dataset_120, and
dataset_128 where 34, 120, and 128 refer to the number of raters in the dataset version. The results from the motivation and simulation sections leverage the Waterloo-IV dataset<sup>1</sup>. The in-depth LB
reproducibility utilizes network traces<sup>2</sup>, video chunks<sup>3</sup>, and experience set<sup>4</sup>.

### Extra code for the LB reproducibility
To support independent subjective tests, the GitHub repository includes a standalone Subjective_assessments folder. This folder provides the code of our
iQoE website and also enables real-world subjective assessments with the code that creates training
and testing experiences. To facilitate new simulations, the repository contains a Synthetic_raters
folder with the code that generates synthetic raters, synthetic experiences, and scores of the
experiences by the synthetic raters.

### Installation
The LA reproducibility involves installation of Python 3.7 with libraries as described in the
requirements.txt file. For the in-depth LB replication, the additional installations include Park<sup>5</sup> and
FFmpeg<sup>6</sup>, with Codex<sup>7</sup> as the recommended FFmpeg version, to generate experiences and
calculate PSNR, SSIM, VMAF, and other metrics. The computation of the quantization parameter
leverages ffmpeg-debug-qp<sup>8</sup>. The creation of video chunks uses MP4Box<sup>9</sup>. The experiments with baseline
model L require installation of BiQPS<sup>10</sup>. Reproduction of our subjective studies from scratch involves
deployment of the iQoE website developed in HTML, JavaScript, and CSS.

### Evaluation and expected results
The repository allocates a separate self-contained folder for reproducing the results of each figure or table on the LA and/or LB levels, with readme.md files providing
specific instructions.

The LA reproducibility of the results in Fig. 2-13 and 15-18 as well as Tables 1 and 2 largely
consists in running the provided Python code on the associated dataset. Fig. 2-4 utilize Waterloo-IV
data. Fig. 5, 6, 8, and 15 and Table 1 rely on the dataset_120 version of our iQoE dataset. Fig. 7
leverages the extended dataset_128 version. The validation of the synthetic raters in Fig. 9 entails
the dataset_34 version of the iQoE dataset. The replication of Fig. 9-13 and 16-18 and Table 2 on the
LA level leverages the data already recorded in our simulations, e.g., scores by the synthetic raters.

For completeness of the LA reproducibility, the GitHub repository includes the .pptx source of
our diagrams in Fig. 1 and supports replication of the screenshots in Fig. 14 by providing access to
our original iQoE website<sup>12</sup> with anonym and iQoE_92 as the username and password, respectively.

Comprehensive replication of our results on the LB level involves additional existing and newly
developed software. The code in the Subjective_assessments folder enables a researcher to recreate
the subjective studies by deploying an own version of our iQoE website and preparing own training
and testing experiences so as to reproduce Fig. 5-9 and 15 and Table 1. The Synthetic_raters folder
empowers the researcher to engender new synthetic raters and conduct independent simulations for
the LB reproducibility of Fig. 9-13 and 16-18 and Table 2.

<sup>1</sup>https://dx.doi.org/10.21227/j15a-8r35

<sup>2</sup>https://doi.org/10.6084/m9.figshare.24460084.v1

<sup>3</sup>https://doi.org/10.6084/m9.figshare.24460078.v2

<sup>4</sup>https://doi.org/10.6084/m9.figshare.24460081.v1

<sup>5</sup>https://github.com/park-project/park

<sup>6</sup>https://ffmpeg.org/download.html

<sup>7</sup>https://www.gyan.dev/ffmpeg/builds

<sup>8</sup>https://github.com/slhck/ffmpeg-debug-qp

<sup>9</sup>https://github.com/gpac/gpac/wiki/MP4Box

<sup>10</sup>https://github.com/TranHuyen1191/BiQPS

<sup>11</sup>https://iqoe.itec.aau.at


### Requirements

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

### Terminology
In the textual descriptions of the repository, as well as in the names of folders and files, we employ terms rater and user interchangeably.
