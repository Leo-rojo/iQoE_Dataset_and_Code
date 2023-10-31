### Generation of IFs for 1000 experiences of 4 chunks each:

#### Description of folder ABR_simulator:
  * The folder 'park' contains the used simulator, the original code can be found in 'https://github.com/park-project/park'. The simulator needs: 1) compliant network traces and 2) sizes of video chunks. We have already inserted the elaborated throughput traces and video_sizes. 
  The necessary traces files and data to generate the used traces can be downloaded from: https://doi.org/10.6084/m9.figshare.24460084.v1
  * Generate_sizes_for_park.py generates video_sizes_ToS.npy to be included in park. It contanins the sizes of the different chunks considered
  * 'agent_class.py' contains implementations of various ABRs.
  
#### Instruction to generate the videostreaming experiences: 
  1. Follow the instruction in features_calculation folder to generate the 10 used features for all the chunks. This will produce a folder called chunk_features that need to be copied in experience_collection folder.
  2. Run 'Simulation.py' to generate three npy files 'exp_bb.npy', 'exp_mpc.npy', 'exp_th.npy' containing videostreaming experience description for different ABRs (from the agent_class.py).
  3. Put the three previous generated files in 'experience_collection' and  run 'map_bit_to_chunk_features.py' to produce the videostreaming experience with features information for each chunk. The produced file is called 'experiences_with_features.npy'.
  4. Run 'generate_1000_shorter_exps.py' to generate the description of 1000 shorter experiences of 4 chunks each. The produced file is called 'iQoE_synth_exp.npy' and it needs to be copied in Video_pools folder.


