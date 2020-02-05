# PyTorch Image Classification TestBench

You can train or test MobileNet/MobileNetV2/ShuffleNet/ShuffleNetV2 on CIFAR10/CIFAR100/ImageNet.  
Specially, you can train or test on any device (CPU/sinlge GPU/multi GPU) and resume on different device environment available.

## Requirements

- python 3.5+
- pytorch 1.0+
- torchvision 0.4+
- numpy
- requests (for downloading pretrained checkpoint and imagenet dataset)


## How to download the ImageNet data

```
usage: down_imagenet.py [-h] [--datapath PATH]

optional arguments:
  -h, --help       show this help message and exit
  --datapath PATH  Where you want to save ImageNet? (default: ../data)
```

### usage

``` shell
$ python3 down_imagenet.py
```

> ***Please check the datapath***  
> Match the same as the datapath argument used by **`main.py`**.

## How to download a pretrained Model

The pretrained models of ShuffleNet and ShuffleNetV2 trained on ImegeNet is not available now..

### Usage

``` shell
$ python down_ckpt.py imagenet -a mobilenet -o pretrained_model.pth
```

***for downloading all checkpoints***

``` shell
$ ./down_ckpt_all.sh
```

----------

## How to train / test networks

```
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [-b N] [--lr LR]
               [--momentum M] [--wd W] [--width-mult WM] [--groups N] [-p N]
               [--ckpt PATH] [-r] [-e] [-C] [-g GPUIDS [GPUIDS ...]]
               [--datapath PATH]
               DATA

positional arguments:
  DATA                  dataset: cifar10 | cifar100 | imagenet (default:
                        cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mobilenet | mobilenetv2 |
                        shufflenet | shufflenetv2 (default: mobilenet)
  -j N, --workers N     number of data loading workers (default: 8)
  --epochs N            number of total epochs to run (default: 200)
  -b N, --batch-size N  mini-batch size (default: 128), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate (defualt: 0.1)
  --momentum M          momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 5e-4)
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  --groups N            number of groups for ShuffleNet (default: 2)
  -p N, --print-freq N  print frequency (default: 50)
  --ckpt PATH           Path of checkpoint for resuming/testing or retraining
                        model (Default: none)
  -R, --resume          Resume model?
  -E, --evaluate        Test model?
  -C, --cuda            Use cuda?
  -g GPUIDS [GPUIDS ...], --gpuids GPUIDS [GPUIDS ...]
                        GPU IDs for using (Default: 0)
  --datapath PATH       where you want to load/save your dataset? (default:
                        ../data)
```

### Training

#### Train one network with a certain dataset

``` shell
$ python main.py cifar10 -a mobilenet -C -g 0 1 2 3 -b 256
```

#### Resume training

``` shell
$ python main.py cifar10 -a mobilenet -C -g 0 1 2 3 -b 256 -R --ckpt ckpt_epoch_50.pth
```

#### Train all networks on every possible datasets

``` shell
$ ./run.sh
```

#### Train all networks on CIFAR datasets

``` shell
$ ./run_cifar.sh
```

### Test

``` shell
$ python main.py cifar10 -a mobilenet -C -g 0 1 2 3 -b 256 -E --ckpt ckpt_best.pth
```

## Delete Checkpoints (without best validation accuracy checkpoint)

``` shell
$ rm -f checkpoint/*/*/ckpt_epoch_*.pth
```

----------

## TODO

- Update other models
- Make TinyImageNet dataloader
- Update ImageNet pretrained model of ShuffleNet/ShuffleNetV2

----------

## References

- [torchvision models github codes](https://github.com/pytorch/vision/tree/master/torchvision/models)
- [MobileNet, ShuffleNet and ShuffleNetV2 Cifar GitHub (unofficial)](https://github.com/kuangliu/pytorch-cifar)
- [MobileNetV2 Cifar GitHub (unofficial)](https://github.com/tinyalpha/mobileNet-v2_cifar10)
- [ShuffleNet and ShuffleNetV2 GitHub (unofficial)](https://github.com/xingmimfl/pytorch_ShuffleNet_ShuffleNetV2)
- [ShuffleNet GitHub (unofficial)](https://github.com/jaxony/ShuffleNet)
- [ShuffleNetV2 GitHub (unofficial)](https://github.com/Randl/ShuffleNetV2-pytorch)
- [PyTorch-CIFAR100 Benchmark list](https://github.com/weiaicunzai/pytorch-cifar100)
