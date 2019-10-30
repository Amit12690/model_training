import shutil
import os 
from glob import glob 

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

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

def copy_imgs_from_xml_fnames(xml_folder , src_img_folder , dest_folder):
    xml_names = glob(xml_folder+'/*.xml')
    
    mkdir(dest_folder)
    
    for xml_name in xml_names :
        basename = xml_name.split('.xml')[0].split('/')[-1]
        src_imname = src_img_folder + '/' +   basename 
        dest_imname = dest_folder + '/' + basename
        if os.path.isfile(src_imname+'.jpg'):
            shutil.copy(src_imname + '.jpg' , dest_imname + '.jpg')
        elif os.path.isfile(src_imname+'.png'):
            shutil.copy(src_imname + '.png' , dest_imname + '.png') 
        else :
            print("%s doesn't exist"%(src_imname))
            
def create_train_test_with_desired_folderpath(desired_folder , img_list_txt):
    
    img_list = get_list_frm_txt(img_list_txt)
    
    new_img_list = []
    
    for impath in img_list :
        new_impath = desired_folder+ '/' + impath.split('/')[-1]
        new_img_list.append(new_impath)
    
    mkdir(desired_folder)
    base_txt_name = img_list_txt.split('/')[-1].split('.txt')[0]
    new_txt_name = desired_folder + '/' + base_txt_name + '_desired.txt'
    write_list_to_txt(new_img_list , new_txt_name)
        
            
