import json
import os
import argparse
import shutil
from datetime import datetime

def __get_list_from_txt(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def __load_json_file(json_file):
    with open(json_file) as f:
        json_data = json.load(f)
    return json_data

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def __backup_data_txt_files(dataset_folder, checkpoint_folder):
    shutil.copy(dataset_folder+'/train.txt', checkpoint_folder)
    shutil.copy(dataset_folder+'/valid.txt', checkpoint_folder)
    shutil.copy(dataset_folder+'/classes.names', checkpoint_folder)


parser = argparse.ArgumentParser()
parser.add_argument("--train_json", type=str, help="json file for training params", required=True)
opt = parser.parse_args()
args = vars(opt)

train_json = args["train_json"]
train_data = __load_json_file(train_json)

epochs = train_data["EPOCHS"]
batch_size = train_data["BATCH_SIZE"]
dataset_folder = train_data["DATASET_FOLDER"]
pretrained_weights = train_data["PRETRAINED_WEIGHTS"]
img_size = train_data["IMG_SIZE"]
n_cpu = train_data["N_CPU"]
multiscale_training = train_data["MULTISCALE_TRAINING"]
checkpoint_interval = train_data["CHECKPOINT_INTERVAL"]
evaluation_interval = train_data["EVALUATION_INTERVAL"]
compute_map = train_data["COMPUTE_MAP"]
#clear_old_ckpt = train_data["CLEAR_OLD_CKPT"]
learning_rate = train_data["LEARNING_RATE"]
decay = train_data["DECAY"]
momentum = train_data["MOMENTUM"]


# Creating yolov3 custom cfg
classes_list = __get_list_from_txt(dataset_folder+'/classes.names')
num_classes = len(classes_list)
out_config_path = "{}/yolov3-custom.cfg".format(dataset_folder)
if os.path.isfile(out_config_path):
    shutil.move(out_config_path, "{}/yolov3-custom_old.cfg".format(dataset_folder))
cmd2run = "bash config/create_custom_model_with_lr_etc.sh {} {} {} {} {}".format(num_classes, learning_rate, decay,
                                                                                 momentum, out_config_path)
print("Executing : {}".format(cmd2run))
os.system(cmd2run)

if os.path.isfile(out_config_path):
    print("Train config file written to : {}".format(out_config_path))
else:
    print("Train config file {} is not written !!! Please check.".format(out_config_path))


model_def = "{}/yolov3-custom.cfg".format(dataset_folder)
data_config = "{}/custom.data".format(dataset_folder)
date_str = datetime.now().strftime('%Y-%B-%d')
time_str = datetime.now().strftime("%H_%M_%S")
checkpoint_folder = "{}/trained_models_{}/{}".format(dataset_folder, date_str, time_str)

mkdir(checkpoint_folder)
shutil.copy(train_json, checkpoint_folder)
shutil.copy(model_def, checkpoint_folder)
shutil.copy(data_config, checkpoint_folder)
__backup_data_txt_files(dataset_folder, checkpoint_folder)

train_str = "python train.py --epochs {} --model_def {}  --data_config {} --pretrained_weights {}  --batch_size {} " \
            "--n_cpu {}  --checkpoint_folder {} ".format(epochs, model_def, data_config, pretrained_weights, batch_size,
                                                         n_cpu, checkpoint_folder)
print("\n\n Training : \n "+train_str)
os.system(train_str)
