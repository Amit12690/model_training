# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:07:01 2019

@author: ab
"""

import numpy as np 
from bs4 import BeautifulSoup
import os 
from glob import glob
import argparse
import shutil
import cv2

def get_image_size(imagepath):
    img = cv2.imread(imagepath)
    h,w = img.shape[0:2]
    return [h,w]

def get_list_frm_txt(txt_file,seperator = '\n'):
    with open(txt_file) as f:
        content_list = f.readlines()
    content_list = [i.strip(seperator) for i in content_list]
    return content_list 

def visualise_img(imgs_list_txt,annotations_path,vis_index=[0]):
    img_list = get_list_frm_txt(imgs_list_txt)
    
    for i in vis_index:
        img_path = img_list[i]
        base_img_name = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path)
        img_h , img_w = img.shape[0:2]
        
        annotations = get_list_frm_txt(annotations_path + '/' +  base_img_name + '.txt')
        for annotation in annotations:
            label , cx , cy , w , h = annotation.split(' ')
            label = int(label)
            
            cx = float(cx)*img_w
            cy = float(cy)*img_h        
            w = float(w)*img_w
            h = float(h)*img_h
            
            x1 = int(cx - int(w/2)) 
            y1 = int(cy - int(h/2)) 
            x2 = int(x1+w)
            y2 = int(y1+h)
            
            cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255), 2) 
        cv2.imshow(str(i),img)
        cv2.waitKey(2)
    import pdb; pdb.set_trace()
        
        

imgs_list_txt = '/home/ab/projects/unleashlive1/AI_dev_scripts/pytorch/transfer_learning/full/yolov3/data/custom/train.txt'
annotations_path = '/home/ab/projects/unleashlive1/AI_dev_scripts/pytorch/transfer_learning/full/yolov3/data/custom/labels'

imgs_list_txt = '/home/ab/projects/unleashlive1/AI_dev_scripts/pytorch/transfer_learning/full/yolov3/test_del/train.txt'
annotations_path ='/home/ab/projects/unleashlive1/AI_dev_scripts/pytorch/transfer_learning/full/yolov3/test_del/labels'

visualisation_indices = [0 , 5 , 9]

visualise_img(imgs_list_txt,annotations_path,visualisation_indices)

