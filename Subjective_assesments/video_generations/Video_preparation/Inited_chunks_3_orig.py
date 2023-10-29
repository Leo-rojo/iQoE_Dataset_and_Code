import os
import platform

#create folders
new_fold='inited_dashed_files_orig'
if not os.path.exists(new_fold):
    # Create a new directory because it does not exist
    os.makedirs(new_fold)

for files in os.listdir('chunks_orig'):
    if files.split('.')[0].endswith('_'):
        if len(files.split('_')[0])==2:
            for i in ["%.3d" % i for i in range(1, 295)]:
                print(i)
                cmd = "type " + 'chunks_orig\\' + files + ' chunks_orig\\' + files[0:2] + '_' + i + '.m4s' " >> " + new_fold + '\\' + files[0:2] + '_' + i + '.mp4'
                os.system(cmd)
        else:
            for i in ["%.3d" % i for i in range(1,295)]:
                cmd = "type " + 'chunks_orig\\'+files + ' chunks_orig\\'+files[0]+'_'+i+'.m4s' " >> " + new_fold+'\\'+ files[0]+'_'+i+'.mp4'
                os.system(cmd)

print('done')