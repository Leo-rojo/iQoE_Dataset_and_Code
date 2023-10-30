import os
if not os.path.exists('./chunks_orig'):
    os.makedirs('./chunks_orig')
ms_chunks=[2000]#[10000,4000,2000]
lista=''
for ms_chunk in ms_chunks:
    for video in os.listdir('./original_video_short'):
        lista+='original_video_short\\'+video+' '
        print(video)
        #fold_name=new_fold+'/'+video.split('.')[0] + '_chunks'+str(ms_chunk)+'/'
        #os.makedirs(fold_name)
    cmd = 'mp4box -dash ' + str(ms_chunk) + ' -profile main -rap -out chunks_orig\\chunkmpd '+ '-segment-name 14_$Number%03d$ '+ lista
    os.system(cmd)
print('done')