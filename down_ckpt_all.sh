#!/usr/bin/env bash

for data in cifar10 cifar100 imagenet
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2
    do
        python3 down_ckpt.py $data -a $model -o ckpt_best.pth
    done
done
