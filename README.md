# Keras Implementation of Deep Capsule Neural Networks
Official Implementation of "DeepCaps: Going Deeper with Capsule Networks" paper presented at CVPR 2019.

This code provides deep capsule neural networks (DeepCaps) implemented in Keras with Tensorflow backend. The code supports training the model in multiple GPUs. 

The current `test error on CIFAR10= 7.26%`.   
---

<p align="center">
  <img align="middle" src="./assets/resnet_0_viz.png" alt="Discrete-depth network" width="240" height="330" />
  <img align="middle" src="./assets/odenet_0_viz.png" alt="Continuous-depth network" width="240" height="330" />
</p>

## Usage
```
git clone https://github.com/brjathu/deepcaps.git
cd deepcaps
```

## Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```
The goal of an ODE solver is to find a continuous trajectory satisfying the ODE that passes through the initial condition.


### Keyword Arguments
 - `rtol` Relative tolerance.
 - `atol` Absolute tolerance.
 - `method` One of the solvers listed below.

#### List of ODE Solvers:

Adaptive-step:
 - `dopri5` Runge-Kutta 4(5) [default].
 - `adams` Adaptive-order implicit Adams.

Fixed-step:
 - `euler` Euler method.
 - `midpoint` Midpoint method.
 - `rk4` Fourth-order Runge-Kutta with 3/8 rule.
 - `explicit_adams` Explicit Adams.
 - `fixed_adams` Implicit Adams.

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
