<h1 align='center'>Segmentation_Factory</h1>

### This is a warehouse for segmentation models, can be used to train your image-datasets for segmentation tasks.
## Preparation
### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### [Download Datasets(including VOC, ADE20K, COCO, COCOStuff, Hubmap, Synapse, CityScapes)](https://pan.baidu.com/s/1LLyIlP3sjuoFAwTBaYflRQ?pwd=0615)

## Project Structure
```
├── datasets: Load datasets
    ├── ade.py: build ade20K dataset
    ├── cityscapes.py: class of cityscapes
    ├── cocostuff.py: class of cityscapes
    ├── extra_transforms.py: image data aug methods
    ├── visualize.py: visualize a batch of valid data(original RGB image & mask)
    ├── voc.py: class of voc
├── models: Segmentation Factory Models
    ├── backbones: Construct backbone modules for feature extractor(convnext, convnextv2, mit, mobilenetv2/v3/v4... etc)
    ├── heads: Construct segmentation heads(fpn, segformer, upernet, etc)
    ├── layers: Construct "drop_path", "conv_moudle", "trunc_normal_" layers.
    ├── modules: Construct "ppm", "psa" modules.
├── scheduler:
    ├──scheduler_main.py: Fundamental Scheduler module
    ├──scheduler_factory.py: Create lr_scheduler methods according to parameters what you set
    ├──other_files: Construct lr_schedulers (cosine_lr, poly_lr, multistep_lr, etc)
├── util:
    ├── utils.py: Record various indicator information and output and distributed environment
    ├── losses.py: Define loss functions('CrossEntropy', 'OhemCrossEntropy', 'Dice', 'FocalLoss', etc)
    ├── metrics.py: Define Metrics (pixel_acc, f1score, miou)
├── engine.py: Function code for a training/validation process
├── estimate_model.py: Visualized of pretrained model in validset
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___,  ___data_set___, ___batch_size___, ___num_workers___, ___backbone___, ___seg_head___ and ___nb_classes___ parameters.  

## Train this model
### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.fintune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)


### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will get the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error.  

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
Using a specified part of the GPUs: for example, I want to use the ___second___ and ___fourth___ GPUs:  
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
For the specific number of GPUs on each machine, modify the value of ___--nproc_per_node___.  
If you want to specify a certain GPU, just add ___CUDA_VISIBLE_DEVICES=___ to specify the index number of the GPU before each command.  
The principle is the same as single-machine multi-GPU training:  

#### <1>On the first machine:

```
python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

#### <2>On the second machine: 

```
python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```



## ONNX Deployment

### step 1: ONNX export
```bash
python onnx_export.py
```

### step2: ONNX optimise
```bash
python onnx_optimise.py
```

### step3: ONNX validate
```bash
python onnx_validate.py
```

## Citation

```
@article{qin2024mobilenetv4,
  title={MobileNetV4-Universal Models for the Mobile Ecosystem},
  author={Qin, Danfeng and Leichner, Chas and Delakis, Manolis and Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and Banbury, Colby and Ye, Chengxi and Akin, Berkin and others},
  journal={arXiv preprint arXiv:2404.10518},
  year={2024}
}
```

```
@inproceedings{woo2023convnext,
  title={Convnext v2: Co-designing and scaling convnets with masked autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Kweon, In So and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16133--16142},
  year={2023}
}
```

```
@article{wang2023crossformer++,
  author       = {Wenxiao Wang and Wei Chen and Qibo Qiu and Long Chen and Boxi Wu and Binbin Lin and Xiaofei He and Wei Liu},
  title        = {Crossformer++: A versatile vision transformer hinging on cross-scale attention},
  journal      = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence, {TPAMI}},
  year         = {2023},
  doi          = {10.1109/TPAMI.2023.3341806},
}
```

```
@inproceedings{liu2022convnet,
  title={A convnet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11976--11986},
  year={2022}
}
```

```
@article{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={12077--12090},
  year={2021}
}
```

```
@inproceedings{wang2021crossformer,
  title = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention},
  author = {Wenxiao Wang and Lu Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
  booktitle = {International Conference on Learning Representations, {ICLR}},
  url = {https://openreview.net/forum?id=_PHymLIxuI},
  year = {2022}
}
```

```
@inproceedings{howard2019searching,
  title={Searching for mobilenetv3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1314--1324},
  year={2019}
}
```

```
@inproceedings{xiao2018unified,
  title={Unified perceptual parsing for scene understanding},
  author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={418--434},
  year={2018}
}
```

```
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

```
@inproceedings{lin2017feature,
  title={Feature pyramid networks for object detection},
  author={Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2117--2125},
  year={2017}
}
```

```
@inproceedings{zhao2017pyramid,
  title={Pyramid scene parsing network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2881--2890},
  year={2017}
}
```

