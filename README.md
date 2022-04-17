# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NIghFeKQGFRVRPh553JaQXeIMlI1xh95#)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-learn-one-representation-unified/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=you-only-learn-one-representation-unified)

![Unified Network](https://github.com/WongKinYiu/yolor/blob/main/figure/unifued_network.png)

<img src="https://github.com/WongKinYiu/yolor/blob/main/figure/performance.png" height="480">

To get the results on the table, please use [this branch](https://github.com/WongKinYiu/yolor/tree/paper).

## Training Preparations

Install YOLOR dependencies
<details><summary> <b>Expand</b> </summary>

* clone YOLOR repository
      
      !git clone https://github.com/Setiawanfikri/YOLOR
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

Training schedule in the paper:

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg cfg/yolor_p6.cfg --weights '' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolor_p6 --hyp hyp.scratch.1280.yaml --epochs 300
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 tune.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg cfg/yolor_p6.cfg --weights 'runs/train/yolor_p6/weights/last_298.pt' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolor_p6-tune --hyp hyp.finetune.1280.yaml --epochs 450
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg cfg/yolor_p6.cfg --weights 'runs/train/yolor_p6-tune/weights/epoch_424.pt' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolor_p6-fine --hyp hyp.finetune.1280.yaml --epochs 450
```

## Evaluate Custom YOLOR Detector Performance

[`yolor_p6.pt`](https://drive.google.com/file/d/1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76/view?usp=sharing)

```
python detect.py --source inference/images/horses.jpg --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --conf 0.25 --img-size 1280 --device 0
```

You will get the results:

![horses](https://github.com/WongKinYiu/yolor/blob/main/inference/output/horses.jpg)

## Citation

```
@article{wang2021you,
  title={You Only Learn One Representation: Unified Network for Multiple Tasks},
  author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2105.04206},
  year={2021}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

</details>
