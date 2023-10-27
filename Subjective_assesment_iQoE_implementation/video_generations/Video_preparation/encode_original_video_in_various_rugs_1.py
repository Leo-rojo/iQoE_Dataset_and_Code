from os import path, listdir, makedirs
from subprocess import call
import sys
import shutil
import time
import os
chunk_length_in_sec=2
def encode_parall(dimensions, bitrate):
    inputDirectory = './original_video_short/'  # to check if it ends in a /, and if not add the / to it.
    filesInInputDirectory = [inputDirectory + fileName for fileName in listdir(inputDirectory)]
    outputDirectory = './encoded_video/'
    # output video framerate
    FRAMERATE = '24'
    FREQ_OF_CHECKS=1.5
    # encoding speed:compression ratio
    PRESET = 'fast'
    # output file format
    OUTPUT_FILE_EXTENSION = 'mp4'
    # relative output directory
    RELATIVE_OUTPUT_DIRECTORY = 'encoded'
    # ffmpeg path
    FFMPEG_PATH = 'ffmpeg'  # '/usr/bin/ffmpeg'
    call([
        FFMPEG_PATH,
        '-i', filesInInputDirectory[0],
        '-f', OUTPUT_FILE_EXTENSION,
        '-s', dimensions,
        '-b:v', bitrate,
        '-minrate', bitrate,
        '-maxrate', bitrate,
        '-bufsize', str(float(bitrate[:-1])/FREQ_OF_CHECKS)+'k',
        '-r', FRAMERATE,
        '-vcodec', 'libx264',
        '-x264-params', "nal-hrd=cbr",
        '-g', str(int(FRAMERATE)*chunk_length_in_sec),#'48', #24*2 #every 48 frames I want a keyframe for chunkify later
        '-keyint_min', str(int(FRAMERATE)*chunk_length_in_sec/2),#'24', #24*2/2
        '-force_key_frames', "expr:gte(t,n_forced*"+str(chunk_length_in_sec)+")", #2 are the seconds
        #'-preset', PRESET,
        '-threads', '0',
        outputDirectory + 'dim_'+dimensions+'__bit' + '_' + bitrate +'.mp4'
    ])

if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists('./encoded_video'):
        os.makedirs('./encoded_video')
    outputDirectory = './encoded_video/'
    #clean_directory(outputDirectory)


    # output video dimensions
    DIMENSIONS = ['3840x2160','3840x2160','2560x1440','1920x1080','1920x1080','1280x720','1280x720','960x520','640x360',
                 '512x288','512x288','384x216','320x180']
    # controls the approximate bitrate of the encode
    BITRATES = ['16800k','11600k','8100k','5800k','4300k','3000k','2350k','1750k','1050k','750k','560k','375k','235k']
    encoding_params=[(DIMENSIONS[x], BITRATES[x]) for x in range(len(BITRATES))]
    time_1 = time.time()
    #encode_parall(DIMENSIONS[0],BITRATES[0])
    with Pool() as p:
        p.starmap(encode_parall,encoding_params) #encoding_params)
    p.close()
    p.join()
    time_2=time.time()
    time_interval = time_2 - time_1
    print(time_interval)
