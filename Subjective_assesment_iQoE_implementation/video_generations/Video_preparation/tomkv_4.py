import os
new_fold='mkvfiles'
if not os.path.exists(new_fold):
    # Create a new directory because it does not exist
    os.makedirs(new_fold)
for files in os.listdir('inited_dashed_files'):
    path_video='inited_dashed_files\\'+files
    path_i_video='mkvfiles\\'+files.split('.')[0]+'.mkv'
    komanda = "ffmpeg -fflags +genpts -i " + path_video + " -c copy " + path_i_video
    os.system(komanda)
print('done')