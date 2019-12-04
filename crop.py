# -*- coding:utf-8 -*-  

import argparse
import cv2
import dlib
import json
import numpy
import skimage
from pathlib import Path
from tqdm import tqdm
from umeyama import umeyama
from os.path import join
import json
import os
import subprocess

from face_alignment import FaceAlignment, LandmarksType
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
def monkey_patch_face_detector(_):

    detector = dlib.get_frontal_face_detector()
    class Rect(object):
        def __init__(self,rect):
            self.rect=rect
    def detect( *args ):
        return [ Rect(x) for x in detector(*args) ]
    return detect

dlib.cnn_face_detection_model_v1 = monkey_patch_face_detector
#FACE_ALIGNMENT = FaceAlignment( LandmarksType._2D, enable_cuda=True, flip_input=False )
FACE_ALIGNMENT = FaceAlignment( LandmarksType._2D, device='cuda', flip_input=False )


mean_face_x = numpy.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = numpy.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = numpy.stack( [ mean_face_x, mean_face_y ], axis=1 )

def transform( image, mat, size, padding=0 ):
    mat = mat * size
    mat[:,2] += padding
    new_size = int( size + padding * 2 )
    return cv2.warpAffine( image, mat, ( new_size, new_size ) )
    
def iter_face_alignments(input_files,only_one_face,input_dir,output_dir):
    for fn in tqdm( input_files ):
     
        #print('--------------- 3')
        image = cv2.imread( str(fn) )
        #print('--------------- 4')

        if image is None:
          # tqdm.write( "Can't read image file: ", fn )
          continue

        # ar=image.shape[1]/image.shape[0]
        # image = cv2.resize(image, (int(800*ar),800),cv2.INTER_CUBIC)
        #print('--------------- 4.5')
        faces = FACE_ALIGNMENT.get_landmarks( image.copy() )
        #print('--------------- 5')

        if faces is None:
            #print(fn) 
            continue
        if len(faces) == 0: 
            #print(fn)
            continue
        if only_one_face and len(faces) != 1: 
            #print(fn)
            continue
        #print('--------------- 6')

        for i, points in enumerate(faces):
            #print('--------------- 7')
                    
            alignment = umeyama( points[17:], landmarks_2D, True )[0:2]
            aligned_image = transform( image, alignment, 160, 48 )
            #print('--------------- 8')

            if len(faces) == 1:
                out_fn = "{}.png".format( Path(fn).stem )
            else:
                out_fn = "{}_{}.png".format( Path(fn).stem, i )
            #print('--------------- 9')

            out_fn = output_dir / out_fn
            #print('+++++++++',out_fn)
            cv2.imwrite( str(out_fn), aligned_image )
            #print('--------------- 10')

            yield str(fn.relative_to(input_dir)), str(out_fn), list( alignment.ravel() ), list(points.flatten().astype(float))
            #print('--------------- 11')


f=open("/home/hzh/dfdc/fb_dfd_release_0.1_final/dataset.json")

#f=open("/home/hzh/dataset_2.json")
dataset=json.load(f)

doc=open('dataset_10fps_crop_train.txt','w')
doc_2=open('dataset_10fps_crop_test.txt','w')

for key in tqdm(dataset):
    label=key.split('/')[0]
    
    set=dataset[key]['set']
    if label=='method_A':
        label_2=0
    elif label=='method_B':
        label_2=1
    elif label=='original_videos':
        label_2=2
        
    if label_2==1 and key.split('/')[1]=='2040724':
        print('method_B 2040724 is none!')
        continue
        
    #print('----------------',label_2)
    '''
    aug=dataset[key]['augmentations']
    if len(aug):
        print(aug)
        if aug[0]=="low_res":
            low_res+=1
        elif aug[0]=="low_quality":
            low_quality+=1
        elif aug[0]=="low_fps":
            low_fps+=1
    '''

    video_path='/home/hzh/dfdc_frames_10fps/'+key.split('.')[0]
    #output_path='/home/hzh/dfdc_frames_10fps_crop/'+key.split('.')[0]
    output_path='/home/hzh/dfdc_10fps_crop/'+key.split('.')[0]
    #print('---------------',output_path)
    #os.makedirs(output_path,exist_ok=True)
    isExist=os.path.exists(output_path)
    if not isExist:
        #print('666666')
        os.makedirs(output_path)
    input_dir=Path(video_path)
    output_dir=Path(output_path)
    output_file = input_dir/'alignments.json'
    input_files = list( input_dir.glob( "*." + 'png' ) )
    
    face_alignments = list( iter_face_alignments(input_files,True,input_dir,output_dir) )
    if set=='train':
        print(output_path,' ',label_2,file=doc)
    elif set=='test':
        print(output_path,' ',label_2,file=doc_2)
    #with output_file.open('w') as f:
        #results = json.dumps( face_alignments, ensure_ascii=False )
        #f.write( results )

    #print( "Save face alignments to output file:", output_file )
