import os

ms_chunks=[2000]#[10000,4000,2000]
lista=''
for ms_chunk in ms_chunks:
    for video in os.listdir('./encoded_video'):
        lista+='encoded_video\\'+video+' '
        print(video)
        #fold_name=new_fold+'/'+video.split('.')[0] + '_chunks'+str(ms_chunk)+'/'
        #os.makedirs(fold_name)
    cmd = 'mp4box -dash ' + str(ms_chunk) + ' -profile main -rap -out chunks\\chunkmpd '+ '-segment-name $RepresentationID$_$Number%03d$ '+ lista
    os.system(cmd)
print('done')