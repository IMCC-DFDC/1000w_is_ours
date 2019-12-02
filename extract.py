#-*- coding: UTF-8 -*-
import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm


def extract_frames(data_path, output_path, method='ffmpeg'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} -r 1 {}'.format(
                data_path, join(output_path, output_path.split('/')[-1]+'_%04d.png')),
            shell=True, stderr=subprocess.STDOUT)

if __name__=='__main__':
    #data_path='/home/hzh/dfdc/fb_dfd_release_0.1_final/original_videos'
    data_path = '/home/hzh/dfdc/fb_dfd_release_0.1_final/method_A'

    #print(os.listdir(data_path))

    for path_2 in os.listdir(data_path):
        path_22=join(data_path,path_2)
        for path_3 in os.listdir(path_22):
            path_33=join(path_22,path_3)
            for video in tqdm(os.listdir(path_33)):
                video_path=join(data_path,path_2,path_3,video)
                output_path=join(data_path+'_frames',path_2,path_3,video.split('.')[0])
                isExist=os.path.exists(output_path)
                if not isExist:
                    os.makedirs(output_path)
                extract_frames(video_path,output_path)

    '''

    #对原视频解帧
    for path_2 in os.listdir(data_path):
        path_22=join(data_path,path_2)
        for video in tqdm(os.listdir(path_22)):
            video_path=join(data_path,path_2,video)
            output_path=join(data_path+'_frames',path_2,video.split('.')[0])
            isExists=os.path.exists(output_path)
            if not isExists:
                os.makedirs(output_path)
            extract_frames(video_path,output_path)
            
                reader=cv2.VideoCapture(video_path)
                frame_num=0
                while reader.isOpened():
                    success,image=reader.read()
                    if not success:
                        print('VideoCapture fail!')
                        break
                    cv2.imwrite(join(output_path,'{:05d}.png'.format(frame_num)),image)
                    frame_num+=1
                reader.ralease()
            
            
                subprocess.check_output('ffmpeg - i {} {}'.format(video_path,join(output_path,'%04d.png')),
                                        shell=True,stderr=subprocess.STDOUT)
            
        '''









