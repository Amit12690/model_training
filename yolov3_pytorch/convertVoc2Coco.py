import numpy as np 
from bs4 import BeautifulSoup
import os 
from glob import glob
import argparse
import shutil
import cv2
import random
SEED = 11
random.seed(SEED)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def copy(src, dest):
    shutil.copy(src, dest)   

def get_image_size(imagepath):
    img = cv2.imread(imagepath)
    h,w = img.shape[0:2]
    return [h,w]

def get_annotations_yolo_format(label_id , bbox_ltrb, img_height , img_width):
    """
    input bbox expected in [y1,x1,y2,x2] format ie., [ top , left , , bottom , right  ] format 
    output --> [centroid_x , centroid_y , w , h]
    """
    bh = bbox_ltrb[2]-bbox_ltrb[0]
    bw = bbox_ltrb[3]-bbox_ltrb[1]
    cy = (bbox_ltrb[2]+bbox_ltrb[0])/2.0 #centroid x
    cx = (bbox_ltrb[3]+bbox_ltrb[1])/2.0 #centroid y
    bbox_ccwh = [cx,cy ,bw , bh]
    bbox_ccwh = np.array(bbox_ccwh)/[img_width, img_height ,img_width, img_height ]
    yolo_format_ann = str(label_id) + ' ' + \
                      str(bbox_ccwh[0]) + ' ' + \
                      str(bbox_ccwh[1]) + ' ' + \
                      str(bbox_ccwh[2]) + ' ' + \
                      str(bbox_ccwh[3])
    return yolo_format_ann
                      

def get_annotations_as_dict(xml_bs):
    objectwise_bbox_dict = {}
    ann_obj = xml_bs
    objs = ann_obj.findAll('object')
    for obj in objs:
        obj_names = obj.findChildren('name')
        for name_tag in obj_names:
            #if str(name_tag.contents[0]) in classes :
            object_name = str(name_tag.contents[0])
            if object_name not in objectwise_bbox_dict.keys():
                objectwise_bbox_dict[object_name] = []            
            fname = ann_obj.findChild('filename').contents[0]
            bbox = obj.findChildren('bndbox')[0]
            xmin = int(bbox.findChildren('ymin')[0].contents[0])
            ymin = int(bbox.findChildren('xmin')[0].contents[0])
            xmax = int(bbox.findChildren('ymax')[0].contents[0])
            ymax = int(bbox.findChildren('xmax')[0].contents[0])
            objectwise_bbox_dict[object_name].append([xmin, ymin, xmax, ymax])
    return objectwise_bbox_dict

def get_bboxes_from_annotation(xml_file):
    """
    Load annotation file for a given image.
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        BeautifulSoup structure: the annotation labels loaded as a
            BeautifulSoup data structure
    """
    xml = ""
    with open(xml_file) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    xml_bs = BeautifulSoup(xml)
    
    annotation_dict = get_annotations_as_dict(xml_bs)
    
    return annotation_dict
    
def get_list_frm_txt(txt_file,seperator = '\n'):
    with open(txt_file) as f:
        content_list = f.readlines()
    content_list = [i.strip(seperator) for i in content_list]
    return content_list 

def write_list_to_txt(content_list , out_name):
    f=open(out_name,"w")
    for i in content_list:
        f.write(i+'\n')
    f.close()
    
def get_train_val_split(img_list,split_factor = 0.7):
    total_files = len(img_list)
    random.shuffle(img_list)
    tr_num = int(round(total_files * split_factor))

    tr_files = img_list[0:tr_num]
    val_files = img_list[tr_num:]
    return  tr_files , val_files   
    
def write_yolo_annotations(annotation_dict , img_list_out_path , annotations_out_path):
    img_list = []
    for imgpath in list(annotation_dict.keys()):
        img_list.append(imgpath)
        base_img_name = imgpath.split('/')[-1].split('.')[0]
        ann_fname = annotations_out_path+'/'+base_img_name+'.txt'
        ann_list = []
        for ann in annotation_dict[imgpath]:
         ann_list.append(ann)
        write_list_to_txt(ann_list , ann_fname)
    write_list_to_txt(img_list , img_list_out_path)
                     
    
def get_label_ID_map(label_map_txt):
    labels= get_list_frm_txt(label_map_txt)
    label_Ids = np.arange(len(labels))
    label_ID_map = dict(zip(labels,label_Ids))
    return label_ID_map
    
def convertVoc2Yolo(xml_folder, images_list , label_map_txt):
    xml_list = glob(xml_folder+'/*.xml')
    
    label_id_map = get_label_ID_map(label_map_txt)
    
    yolo_annotation_dict = {} 
    for imagefile in images_list:
       yolo_annotation_list = []
       base_image_name = imagefile.split('/')[-1].split('.')[0]
       if os.path.isfile(imagefile):
           yolo_image_file  = imagefile
           img_height , img_width=get_image_size(yolo_image_file)
       else :
           yolo_image_file = None

       if yolo_image_file != None :
           xml_file = xml_folder + '/' + base_image_name + '.xml'
           if xml_file in xml_list:
               annotation_dict = get_bboxes_from_annotation(xml_file)
               #person': [[577, 254, 679, 353], [337, 853, 403, 877], [325, 882, 389, 914]]
               for label in annotation_dict.keys():
                   label_id = label_id_map[label]
                   for bbox in annotation_dict[label]:                       
                       yolo_ann = get_annotations_yolo_format(label_id , bbox, img_height , img_width)
                       yolo_annotation_list.append(yolo_ann)
                       #yolo_imgs_list.append(yolo_image_file)
               if len(yolo_annotation_list) > 0 : 
                  yolo_annotation_dict[yolo_image_file] = yolo_annotation_list
    return yolo_annotation_dict
           
      

parser = argparse.ArgumentParser()
parser.add_argument("--xml_folder", type=str, help="xml folder")
parser.add_argument("--images_folder", type=str , help="images folder")   
parser.add_argument("--label_map_txt" , type=str , help= "label map txt file")
parser.add_argument("--out_folder", type=str , help="folder to save the annotations")
parser.add_argument("--train_val_split", type=float , default = 0.8 , help="train-validation split percentage")
#parser.add_argument("--train_txt" , type=str , default = None ,help= "train_img_list")
#parser.add_argument("--valid_txt" , type=str , default = None , help= "valid_img_list") 
opt = parser.parse_args()
args = vars(opt)

images_folder = args['images_folder']
out_folder = args['out_folder']
train_val_split = args['train_val_split']
mkdir(out_folder)

images_list = list(glob(images_folder+'/*.jpg')) + list(glob(images_folder+'/*.png'))
tr_list , val_list = get_train_val_split(images_list,split_factor = train_val_split)

if tr_list != None :
   yolo_tr_annotations = convertVoc2Yolo(args['xml_folder'], tr_list , args['label_map_txt']) 
   mkdir(out_folder+'/labels')
   write_yolo_annotations(yolo_tr_annotations, img_list_out_path = out_folder+'/train.txt' , annotations_out_path = out_folder+'/labels' )
   
if val_list != None :
   yolo_val_annotations = convertVoc2Yolo(args['xml_folder'], val_list , args['label_map_txt']) 
   mkdir(out_folder+'/labels')
   write_yolo_annotations(yolo_val_annotations , img_list_out_path = out_folder+'/valid.txt' , annotations_out_path = out_folder+'/labels')
   