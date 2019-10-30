import os
import argparse

"""
Make sure the dataset folder has this format : 
<dataset_folder>
|_ images
|
|_ annotations
|  |_ xmls
|
|_ classes.names
"""


def __get_list_from_txt(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def create_custom_data_config(dataset_out_folder, label_map_txt):
    classes_list = __get_list_from_txt(label_map_txt)
    num_classes = len(classes_list)
    train_list_path = dataset_out_folder + '/train.txt'
    val_list_path = dataset_out_folder + '/valid.txt'

    error = False
    if not os.path.isfile(train_list_path):
        print("{} doesn't exist. Please check".format(train_list_path))
        error = True
    if not os.path.isfile(val_list_path):
        print("{} doesn't exist. Please check".format(val_list_path))
        error = True

    assert not error, "\nPlease check the errors above"

    str2write = "classes={}\ntrain={}\nvalid={}\nnames={}".format(num_classes,train_list_path,
                                                                  val_list_path,label_map_txt)
    custom_data_config = dataset_out_folder+'/custom.data'
    with open(custom_data_config, "w") as f:
        f.write(str2write)

    assert os.path.isfile(dataset_out_folder+'/custom.data'), "{} not created".format(custom_data_config)
    "{} written successfully".format(custom_data_config)

    return classes_list

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, help="dataset folder", required=True)
parser.add_argument("--out_folder", default=None, type=str, help="folder to save the annotations")
parser.add_argument("--train_val_split", type=float, default=0.8, help="train-validation split percentage")
opt = parser.parse_args()
args = vars(opt)

####### DATASET RELATED
dataset_folder = args['dataset_folder']
out_folder = args['out_folder'] or dataset_folder
train_val_split = args['train_val_split']
xml_folder = dataset_folder + '/annotations/xmls'
images_folder = dataset_folder + '/images' 
label_map_txt = dataset_folder + '/classes.names'

s1 = os.path.isdir(xml_folder)
s2 = os.path.isdir(images_folder)
s3 = os.path.isfile(label_map_txt)
error = False

if not s1:
    print("\nxml_folder not found : {} !!! Please check", xml_folder)
    error = True
if not s2:
    print("\nimages_folder not found : {} !!! Please check", xml_folder)
    error = True
if not s3:
    print("\nlabel_map_txt not found : {} !! Please check ", label_map_txt)
    error = True
assert not error, "\nPlease check the errors above"

cmd2run = "python convertVoc2Coco.py " + "--xml_folder {} ".format(xml_folder) +\
          " --images_folder {} ".format(images_folder) +\
          " --label_map_txt {} ".format(label_map_txt) +\
          " --train_val_split {} ".format(train_val_split) +\
          " --out_folder {}".format(out_folder)
print(cmd2run)
os.system(cmd2run)

# Creating custom.data  config file
classes_list = create_custom_data_config(out_folder, label_map_txt)