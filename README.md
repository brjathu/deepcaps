# Keras Implementation of Deep Capsule Neural Networks
Official Implementation of "DeepCaps: Going Deeper with Capsule Networks" paper presented at CVPR 2019.

This code provides deep capsule neural networks (DeepCaps) implemented in Keras with Tensorflow backend. The code supports training the model in multiple GPUs. 

The current `test error on CIFAR10= 7.26%`.   

## Usage
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
 
## Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```
Commnet the required datasets in the `train.py` file.


## Performance

## Download pre-trained models and ensemble test



### References
[1] J. Rajasegaran, V. Jayasundara, S.Jeyasekara, N. Jeyasekara, S. Seneviratne, R. Rodrigo.. "DeepCaps : Going Deeper with Capsule Networks." *Conference on Computer Vision and Pattern Recognition.* 2019. [[arxiv]](https://arxiv.org/abs/1806.07366)

---

If you found this code useful in your research, please consider citing
```
@article{chen2018neural,
  title={DeepCaps : Going Deeper with Capsule Networks},
  author={Jathushan Rajasegaran, Vinoj Jayasundara, Sandaru Jayasekara, Hirunima Jayasekara, Suranga Seneviratne, Ranga Rodrigo},
  journal={Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
