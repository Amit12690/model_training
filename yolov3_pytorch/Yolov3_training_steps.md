### Pre-requisites for training Yolov3 - pytorch 

#### 1. The training repo (scripts)  

- For yolov3 , clone the repo : https://github.com/Amit12690/model_training.git [branch : master]

- The requirements to be installed are as follows :

  - python-3.6
  - numpy
  - torch>=1.0
  - torchvision
  - matplotlib
  - tensorflow
  - tensorboard
  - terminaltables
  - pillow
  - tqdm

  The same has been provided in **requirements.txt**  so the following should work : 

  > $  pip install -r requirements.txt

- Navigate to  `model_training/yolov3_pytorch`. All the necessary codes are in the same path.



#### 2. The dataset in a specific format 

- Prepare a text file called **classes.names**. This is a simple txt file where class labels are listed in order of the class ids. For example for a vehicle detector if the class IDs  of car=1, truck=2 and bus=3 , the classes.names will look like this :  

  ```reStructuredText
  car
  truck
  bus
  ```

  Example : https://drive.google.com/open?id=1TM_G61XrtqIWSt6IGkpKf2CfXHcLVyKV

- Annotate the data using https://github.com/tzutalin/labelImg  in **VOC format** ie., the box coordinates (x1,y1,x2,y2 and class_label) are stored in xml file per image. Follow the same classIds and names as in the  **classes.names**  above.

- ```
  Make sure the dataset is arranged in this folder format : 
  <dataset_folder>
  |_ images
  | |_ img1.jpg
  | |_ img2.jpg
  | |_ ..
  | |_ ..
  | |_ imgn.jpg
  |
  |_ annotations
  |  |_ xmls
  |    |_ img1.xml
  |    |_ img2.xml
  |    |_ ..
  |    |_ ..
  |    |_ imgn.xml
  |
  |_ classes.names
  ```

- For yolov3 training , we need to convert the dataset from VOC format to  COCO format

- The following helps prepare the dataset for Yolov3 by converting the annotations from xml format to yolo desired format : 

```bash
python create_dataset_n_config.py --dataset_folder <path to above dataset_folder>   --train_val_split 0.7
```



Please find the sample dataset in the path :  https://drive.google.com/open?id=1TM_G61XrtqIWSt6IGkpKf2CfXHcLVyKV



#### 3. Download pretrained yolov3 models

- Follow the below steps to download the pretrained models to **weights** folder 

```bash
$ cd weights/
$ bash download_weights.sh
```

- This will download : 
  - yolov3.weights  (We will be using this set of weights for finetuning, for it's accuracy)
  - yolov3-tiny.weights
  - darknet53.conv.74 ( yolo-v3 backend pretrained on ImageNet used only while finetuning)

- Alternativey the pretrained models can also be downloaded from this link :  https://drive.google.com/open?id=1ZbgL_aARJzAcOQv2huKFq4XjAPWom0KJ


#### 4. Create necessary config files

- train_params.json   : Set the necessary parameters for training like learning rate, epochs, datapaths etc. in this json file.

- The json file is in the path :  [link](https://drive.google.com/drive/folders/1dWMf4-LVbncjN9kcMVTBxiJuZXxNl__G)


### Running the training

Once the above steps are completed , the training can be started using : 

```bash
python run_training.py --train_json train_params.json 
```

