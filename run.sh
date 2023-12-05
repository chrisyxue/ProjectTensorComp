#!/bin/bash


# nohup python main.py --epoch 400 --batch_size 512 --p 0 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.1 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.2 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.3 &

# nohup python main.py --epoch 400 --batch_size 512 --p 0 --dataset cifar100 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.1 --dataset cifar100 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.2 --dataset cifar100 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.3 --dataset cifar100 &

# nohup python main.py --epoch 400 --batch_size 512 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.4 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.1 --dataset cifar10 --model tkresnet18 --r_ratio 0.4 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.2 --dataset cifar10 --model tkresnet18 --r_ratio 0.4 &
# nohup python main.py --epoch 400 --batch_size 512 --p 0.3 --dataset cifar10 --model tkresnet18 --r_ratio 0.4 &


# python main.py --epoch 400 --batch_size 512 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.4


# nohup python main.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar100 --model tkresnet18 --r_ratio 0.6 &
# nohup python main.py --epoch 400 --batch_size 1024 --p 0.1 --dataset cifar100 --model tkresnet18 --r_ratio 0.6 &
# nohup python main.py --epoch 400 --batch_size 1024 --p 0.2 --dataset cifar100 --model tkresnet18 --r_ratio 0.6 &
# nohup python main.py --epoch 400 --batch_size 1024 --p 0.3 --dataset cifar100 --model tkresnet18 --r_ratio 0.6 &


# nohup python main.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.6 &
# nohup python main.py --epoch 400 --batch_size 1024 --p 0.1 --dataset cifar10 --model tkresnet18 --r_ratio 0.6 &
# nohup python main.py --epoch 400 --batch_size 1024 --p 0.2 --dataset cifar10 --model tkresnet18 --r_ratio 0.6 &
# nohup python main.py --epoch 400 --batch_size 1024 --p 0.3 --dataset cifar10 --model tkresnet18 --r_ratio 0.6 &

nohup python main_train_num.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model resnet18 &
nohup python main_train_num.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.2 &
nohup python main_train_num.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.4 &
nohup python main_train_num.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.6 &
nohup python main_train_num.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model tkresnet18 --r_ratio 0.8 &
# python main_train_num.py --epoch 400 --batch_size 1024 --p 0 --dataset cifar10 --model resnet18