# Tensorflow and Keras Implementation of Deep Capsule Neural Networks
Official Implementation of "DeepCaps: Going Deeper with Capsule Networks" paper, will be presented at CVPR 2019.

This code provides deep capsule neural networks (DeepCaps) implemented in Keras with Tensorflow backend. The code supports training the model in multiple GPUs. 

The current `test error on CIFAR10 = 7.26%`.   

## Usage
### step 1 : Install dependencies
```
conda install -c anaconda tensorflow-gpu=1.13.1
conda install -c anaconda keras-gpu 
conda install -c anaconda scipy=1.2*
conda install -c conda-forge matplotlib
conda install -c conda-forge pillow
```
### step 2 : Clone the repository
```
git clone https://github.com/brjathu/deepcaps.git
cd deepcaps
```

## Supported Datasets
 - `CIFAR10`
 - `CIFAR100` 
 - `SVHN` 
 - `F-MNIST`
 - `MNIST`
 - `tiny-imagenet`
 
## Training

If you are training on multiple GPUs change the `numGPU` parameter in `args` class in `train.py` file. 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

If you are training on single GPU change the `numGPU` parameter in `args` class in `train.py` file to 1.
```
CUDA_VISIBLE_DEVICES=0 python train.py or python train.py
```
To test with several other datasets commnet out the required dataset in the `train.py` file.


## Performance

Dataset | Test error | 
-------|:-------:|
CIFAR10 | 7.26% |
SVHN |2.44% |
MNIST |0.28% |
FMNIST |5.27% |



## Download pre-trained models and ensemble test

Download this [CIFAR10 - pretrained models](https://drive.google.com/open?id=1Plj-dH4OoSORqWf-23XxToW0X46NdVmR) and extract the files inside model directory. Then run `ensemble.py` file.
```
python ensemble.py
```

## We credit
We have used [this](https://github.com/XifengGuo/CapsNet-Keras) as the base CapsNet implementation. We thank and credit the contributors of this repository.

## Contact
Jathushan Rajasegaran - brjathu@gmail.com  
Discussions, suggestions and questions are welcome!

## References
[1] J. Rajasegaran, V. Jayasundara, S.Jeyasekara, N. Jeyasekara, S. Seneviratne, R. Rodrigo. "DeepCaps : Going Deeper with Capsule Networks." *Conference on Computer Vision and Pattern Recognition.* 2019. [[arxiv]](https://arxiv.org/abs/1904.09546)


---

If you found this code useful in your research, please consider citing
```
 @InProceedings{Rajasegaran_2019_CVPR,
author = {Rajasegaran, Jathushan and Jayasundara, Vinoj and Jayasekara, Sandaru and Jayasekara, Hirunima and Seneviratne, Suranga and Rodrigo, Ranga},
title = {DeepCaps: Going Deeper With Capsule Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
