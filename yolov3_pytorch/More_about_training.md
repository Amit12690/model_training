

#### Types of training 

There are 2 types of network training generally carried out in deeplearning. 

1. Finetuning 

2. Training from scratch 


#### 1. Finetuning 

- Generally the pretrained models are trained on huge public datasets like COCO , ImageNet etc. depending on their applications. 

- Since these models are already trained on most of the generic objects like vehicles, people, animals etc. , it is always beneficial to utilise their pretrained knowledge and add our improvements on it.
- This process of improving the pretrained models using new data without losing the pretrained knowledge is called finetuning. 
- This needs small datasets and also very less number of training epochs (cycles), thereby saving time
- Works well on generic labels like vehicles, people, animals.
- The learning rate is generally set to a smaller value like :  0.0001



#### 2. Training from scratch 

- This involves training of the model without any pretrained info 
- Useful when the labels to be trained are very unique (like power components, defects etc.) , dataset is large (>10000 images or even more)
- Needs higher number of training epochs (cycles) and hence more time. 
- The learning rate is generally set to a bigger value like :  0.01 


