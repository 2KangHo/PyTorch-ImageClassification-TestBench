#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2
    do
        python3 main.py $data -a $model -C -g 0 1 2 3 -j 8 -b 256
        rm -f checkpoint/*/*/*_ckpt_epoch_*.pth
        python3 main.py $data -a $model -j 8 -C -g 0 --ckpt ckpt_best.pth -E
    done
done
