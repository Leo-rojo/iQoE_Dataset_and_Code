## Generate Figure 18:

### LA
* Input data: folder time_over contains the time needed to select and train a new experience instance at different training steps for each synthetic raters, space_ave.npy and space_std.npy which contains the space on disk of iQoE models at different training steps for all the synthetic raters.
* Run generate_figure_18.py to generate figure 18

### LB
* Input data: folder features_generated_experiences and synthetic_raters_scores_for_generated_experiences. Both of them can be generated in Synthetic_raters.
* Run generate_figure_18.py to generate figure 18 and run all the other scripts contained in scripts. Those files produce in output folder:
  * folder mq which contains the saved iQoE models at different training steps for all the synthetic raters
  * files space_ave.npy and space_std.npy which contains the space on disk occupied by iQoE models at different training steps for all the synthetic raters
  * folder time_over which contains the time needed to select and train a new experience instance at different training steps for each synthetic raters.
* N.B. The scripts use parallelization which can consume a lot of memory. In order to run the scripts some minutes are needed. 
* N.B. Folder mq is extreamly big, around 5GB. 
* N.B. Results regarding the time overhead can differ from the ones reported in the paper (results from LA folder) because they are influenced by the particular configuration or context of the machine executing the scripts.