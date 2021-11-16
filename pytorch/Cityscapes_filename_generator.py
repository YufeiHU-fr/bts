import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

images = recursive_glob(rootdir='/home/student/workspace_Yufei/CityScapes/leftImg8bit/trainSSL',suffix=".png")
print('the number of images is ',len(images))
#exit()
for i,img_path in enumerate( images):
    #print(img_path)
    #print(img_path.split('trainSSL')[1].split('/')[2])
    #print(os.path.basename(img_path).split('_leftImg8bit')[0] + '_disparity.png')
    #print( os.path.join('/home/student/workspace_Yufei/CityScapes/Depth/disparity/trainSSL',str(img_path.split('trainSSL')[1].split('/')[1]),str(img_path.split('trainSSL')[1].split('/')[2]),os.path.basename(img_path).split('_leftImg8bit')[0] + '_disparity.png'))
    #exit()
    print('working on {}th img...'.format(i+1))
    gt_path = os.path.join('/home/student/workspace_Yufei/CityScapes/Depth/disparity/trainSSL',img_path.split('trainSSL')[1].split('/')[1],img_path.split('trainSSL')[1].split('/')[2], os.path.basename(img_path).split('_leftImg8bit')[0] + '_disparity.png')
    #print(os.path.join('/home/student/workspace_Yufei/CityScapes/camera/trainSSL',img_path.split('trainSSL')[1].split('/')[1],img_path.split('trainSSL')[1].split('/')[2]))
    #exit()
    camera_path = os.path.join('/home/student/workspace_Yufei/CityScapes/camera/trainSSL',img_path.split('trainSSL')[1].split('/')[1],img_path.split('trainSSL')[1].split('/')[2], os.path.basename(img_path).split('_leftImg8bit')[0] + '_camera.json')
    #img =  cv2.imread(img_path,cv2.IMREAD_UNCHANGED).astype(np.float16)
    #gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float16)
    #json_camera = open(camera_path)
    #print('img_path:',img_path)
    #print('gt_path:',gt_path)
    #print('camera_path:',camera_path)
    #exit()
    try:
        img =  cv2.imread(img_path,cv2.IMREAD_UNCHANGED).astype(np.float16)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float16)
        json_camera = open(camera_path)
        print('img_path:',img_path)
        print('gt_path:',gt_path)
        print('camera_path:',camera_path)
    except:
        continue
    else:
        f=open('./cityscapesdepthv2_train_files_with_gt.txt','a')
        f.write(img_path+' '+gt_path+' '+camera_path)
        f.write('\n')
   # print('img_path:',img_path)
