## OpenCon: Open-world Contrastive Learning
Yiyou Sun, Yixuan (Sharon) Li


This repo contains the reference source code in PyTorch of the OpenCon framework. 
We introduce a new learning framework, open-world contrastive learning (OpenCon). 
OpenCon tackles the challenges of learning compact representations for both known 
and novel classes, and facilitates novelty discovery along the way. 
For more details please check our paper [OpenCon: Open-world Contrastive Learning](https://arxiv.org/abs/2208.02764) 
(TMLR 23). 

### Dependencies

The code is built with following libraries:

- [PyTorch==1.7.1](https://pytorch.org/)

### Usage

##### Get Started

The pretraining weights by SimCLR can be downloaded in this [link](https://drive.google.com/file/d/19tvqJYjqyo9rktr3ULTp_E33IqqPew0D/view?usp=sharing)
(provided by [ORCA](https://github.com/snap-stanford/orca)).

- To train on CIFAR-100, run

```bash
python main_cifar.py
```

- To train on ImageNet-100, run
```bash
python main_imagenet.py --dataset_root <IMAGENET_ROOT>
```

### Acknowledgement

Our code repo is built based on https://github.com/snap-stanford/orca. Thanks for the great work!

### Citing

If you find our code useful, please consider citing:

```
@inproceedings{
    sun2023opencon,
    title={OpenCon: Open-world Contrastive Learning},
    author={Yiyou Sun and Yixuan Li},
    booktitle={Transactions on Machine Learning Research},
    year={2023},
    url={https://openreview.net/forum?id=2wWJxtpFer}
}
```
