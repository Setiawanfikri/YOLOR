# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NIghFeKQGFRVRPh553JaQXeIMlI1xh95#)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-learn-one-representation-unified/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=you-only-learn-one-representation-unified)

![Unified Network](https://github.com/WongKinYiu/yolor/blob/main/figure/unifued_network.png)

<img src="https://github.com/WongKinYiu/yolor/blob/main/figure/performance.png" height="480">

To get the results on the table, please use [this branch](https://github.com/WongKinYiu/yolor/tree/paper).

This tutorial is based on the YOLOR repository by Wong Kin-Yiu & YOLOR custom dataset training colab notebook by Roboflow. This repository shows training on your own custom objects.

To train custom dataset using YOLOR take the following steps:

* Install YOLOR dependencies
* Download custom YOLOR dataset
* Prepare Pre-Trained Weights
* Custom dataset YOLOR training
* Evaluate YOLOR performance
* Visualize YOLOR training data
* Run YOLOR inference on test images
* Export saved YOLOR weights

## Training Preparations

Install YOLOR dependencies
<details><summary> <b>Expand</b> </summary>

* clone YOLOR repository
      
      !git clone https://github.com/Setiawanfikri/yolor.git
      %cd yolor
      !git reset --hard eb3ef0b7472413d6740f5cde39beb1a2f5b8b5d1
  
* Install necessary dependencies
      
      !pip install -qr requirements.txt
  
* Install Mish CUDA
  
      !git clone https://github.com/JunnYu/mish-cuda
      %cd mish-cuda
      !git reset --hard 6f38976064cbcc4782f4212d7c0c5f6dd5e315a8
      !python setup.py build install
      %cd ..
  
* Install PyTorch Wavelets
      
      !git clone https://github.com/fbcotter/pytorch_wavelets
      %cd pytorch_wavelets
      !pip install .
      %cd ..

</details>

Download custom YOLOR dataset
<details><summary> <b>Expand</b> </summary>
  
* Install Roboflow dependencies
  
      !pip install -q roboflow
      from roboflow import Roboflow
      rf = Roboflow(model_format="yolov5", notebook="roboflow-yolor")
  
* Download Dataset from Roboflow
    
      %cd /content/yolor
      from roboflow import Roboflow
      rf = Roboflow(api_key="85cNlMEKyhhCdduuKla4")
      project = rf.workspace("joseph-nelson").project("uno-cards")
      dataset = project.version(3).download("yolov5")
  
* See YAML category/class
      
      %cat {dataset.location}/data.yaml

</details>

Prepare Pre-trained weight
<details><summary> <b>Expand</b> </summary>

* Get pretrained YOLOR_p6.pt
      
      %cd /content/yolor
      !pip install gdown
      !gdown "https://drive.google.com/uc?id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76"
    
* Prepare YOLOR YAML configuration
      
      import yaml
      with open(dataset.location + "/data.yaml") as f:
          dataMap = yaml.safe_load(f)

      num_classes = len(dataMap['names'])
      num_filters = (num_classes + 5) * 3
      from IPython.core.magic import register_line_cell_magic

      @register_line_cell_magic
      def writetemplate(line, cell):
          with open(line, 'w') as f:
              f.write(cell.format(**globals()))

* Write YAML configuration
      [here](https://github.com/Setiawanfikri/Training/blob/main/YAML%20configuration)
      copy and paste to colab environment

</details>

## Training

Custom dataset training:
 * img: define input image size
 * batch: determine batch size
 * epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
 * data: set the path to our yaml file
 * cfg: specify our model configuration
 * weights: specify a custom path to weights. (Note: We can specify the pretrained weights we downloaded up above with the shell script)
 * name: result names
 * hyp: Define the hyperparamters for training

```
%cd /content/yolor
!python train.py --batch-size 8 --img 416 416 --data {dataset.location}/data.yaml --cfg cfg/yolor_p6.cfg --weights '/content/yolor/yolor_p6.pt' --device 0 --name yolor_p6 --hyp '/content/yolor/data/hyp.scratch.1280.yaml' --epochs 8
```

## Evaluate Custom YOLOR Detector Performance

Evaluate custom YOLOR model
<details><summary> <b>Expand</b> </summary>
 
* Start Tensorboard, run after training is finished. Logs save in runs folder
      
      %load_ext tensorboard
      %tensorboard --logdir runs
  
* Plots data, if tensorboard isn't working
  
      from IPython.display import Image
      from utils.plots import plot_results  # plot results.txt as results.png
      Image(filename='/content/yolor/runs/train/yolor_p6/results.png', width=1000)  # view results.png
  
* Display ground data
      
      print("GROUND TRUTH TRAINING DATA:")
      Image(filename='/content/yolor/runs/train/yolor_p6/train_batch0.jpg', width=900)
      
* Display augmented data
      
      print("AUGMENTED DATA:")
      Image(filename='/content/yolor/runs/train/yolor_p6/train_batch0.jpg', width=900)
      
</details>

Run Inference with trained weight
<details><summary> <b>Expand</b> </summary>
      
* See directory in runs trained folder
    
      %ls runs/train/yolor_p6/weights
      
* Create names file for model
    
      import yaml
      import ast
      with open("/content/yolor/Uno-Cards-3/data.yaml", 'r') as stream:
          names = str(yaml.safe_load(stream)['names'])

      namesFile = open("../data.names", "w+")
      names = ast.literal_eval(names)
      for name in names:
        namesFile.write(name +'\n')
      namesFile.close()
      
* Runs Trained Model with Test Images
      
      !python detect.py --weights "runs/train/yolor_p6/weights/best.pt" --conf 0.5 --source /content/yolor/Uno-Cards-3/test/images --names ../data.names --cfg cfg/yolor_p6.cfg
     
* Display inference on All Test Images
      
      import glob
      from IPython.display import Image, display

      for imageName in glob.glob('/content/yolor/inference/output/*.jpg'): #assuming JPG
          display(Image(filename=imageName))
          print("\n")
      
</details>

## Export Trained Weight to GDrive
* Login to GDrive
    
      from google.colab import drive
      drive.mount('/content/gdrive')
  
* Export Trained Weight
  
      %cp /content/yolor/runs/train/yolor_p6/weights/best.pt /content/gdrive/My Drive
    
## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [WongKinYiu YOLOR Repository](https://github.com/WongKinYiu/yolor)
* [Roboflow YOLOR Colab Notebook](https://colab.research.google.com/drive/1e1Uk9SjxBaagu7aoGZ4oTcqePhnMLM23?usp=sharing)


</details>
