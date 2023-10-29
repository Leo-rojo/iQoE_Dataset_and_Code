## Generate Figure 18:

### LA
* Input data: folder time_over contains the time needed to select and train a new experience instance at different training steps for each synthetic raters, space_ave.npy and space_std.npy which contains the space on disk of iQoE models at different training steps for all the synthetic raters.
* Run generate_figure_18.py to generate figure 18

### LB
* Input data: folder features_generated_experiences and synthetic_raters_scores_for_generated_experiences. Both of them can be generated in Synthetic_raters_implementation.
* Run generate_figure_18.py to generate figure 18 and run all the other scripts contained in scripts. Those files produce in output folder:
  * folder mq which contains the saved iQoE models at different training steps for all the synthetic raters
  * files space_ave.npy and space_std.npy which contains the space on disk occupied by iQoE models at different training steps for all the synthetic raters
  * folder time_over which contains the time needed to select and train a new experience instance at different training steps for each synthetic raters.
* N.B. The scripts use parallelization which can consume a lot of memory. In order to run the scripts some minutes are needed. 
* N.B. folder mq is extreamly big, around 5GB. 