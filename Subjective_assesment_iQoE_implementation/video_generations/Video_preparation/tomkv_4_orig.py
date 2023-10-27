import os
new_fold='mkvfiles_original'
if not os.path.exists(new_fold):
    # Create a new directory because it does not exist
    os.makedirs(new_fold)
for files in os.listdir('inited_dashed_files_orig'):
    path_video='inited_dashed_files_orig\\'+files
    path_i_video='mkvfiles_original\\'+files.split('.')[0]+'.mkv'
    komanda = "ffmpeg -fflags +genpts -i " + path_video + " -c copy " + path_i_video
    os.system(komanda)
print('done')