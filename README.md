# S2Mamba
The official code for the paper "S2Mamba: A Spatial-spectral State Space Model for Hyperspectral Image Classification"  
## Get Started
### Installation
```Shell
pip install -i requirement.txt
```
### Data Preparation
Download HSI datasets and put them into the `./data` directory. For example:
```Shell
  './data/IndianPine.mat'
  './data/Pavia.mat'
  './data/Houston.mat'
  './data/WHU-Hi-LongKou/WHU_Hi_LongKou.mat'
```
### Training and Testing
```Shell
CUDA_VISIBLE_DEVICES=0 python demo_mamba.py --dataset='Indian' --epoches=400 --patches=7 --sess s2mamba --dropout 0.4 --lr 5e-4

CUDA_VISIBLE_DEVICES=0 python demo_mamba.py --dataset='Pavia' --epoches=400 --patches=11 --sess s2mamba --dropout 0.1 --lr 5e-4

CUDA_VISIBLE_DEVICES=0 python demo_mamba.py --dataset='Houston' --epoches=100 --patches=9 --sess s2mamba --dropout 0.1 --lr 1e-4

CUDA_VISIBLE_DEVICES=0 python demo_mamba.py --dataset='WHU_Hi_LongKou' --epoches=400 --patches=9 --sess s2mamba --dropout 0.4 --lr 5e-4

```
## Acknowledgment
Our detection code is built upon [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer) and [Vmamba](https://github.com/MzeroMiko/VMamba). We are very grateful to all the contributors to these codebases.
