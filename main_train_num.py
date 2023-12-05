import torch
import numpy as np
import random
from cifar10 import load_cifar10_with_noise, load_cifar10_with_noise_train_num
from cifar100 import load_cifar100_with_noise
from models.resnet import make_resnet18k
from models.tk2_resnet_new import make_tkresnet18k
# from models.tt_resnet_new import make_ttresnet18k
import pandas as pd
import os
import argparse
import pdb

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
          images, labels = data
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, trainloader, testloader, epochs):
     criterion = torch.nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
     
     train_error_lst = []
     test_error_lst = [] 
     epoch_lst = []

     for epoch in range(epochs):
          for i, data in enumerate(trainloader):
             inputs, labels = data
             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
             optimizer.zero_grad()
             outputs = model(inputs)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()
            #  print(f'{i}/{len(trainloader)}')
          
          epoch_check = 40
          if epoch % epoch_check == 0 or epoch == epochs - 1:
              print(epoch)
              print('Epoch: %d | Loss: %.3f' % (epoch, loss.item()))
              train_acc = evaluate_model(model, trainloader)
              eval_acc = evaluate_model(model, testloader)
              print('Train acc: %.3f | Eval acc: %.3f' % (train_acc, eval_acc))
              train_error_lst.append(100 - train_acc)
              test_error_lst.append(100 - eval_acc)
              epoch_lst.append(epoch)
     return train_error_lst, test_error_lst, epoch_lst 

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--epoch', type=int,default=400)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--p', type=float, default=0)
parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10','cifar100'])
parser.add_argument('--model', type=str, default='resnet18',choices=['resnet18','tkresnet18','ttresnet18'])
parser.add_argument('--r_ratio', type=float, default=0.5)
# parser.add_argument('--train_num', type=int, default=100)
args = parser.parse_args()

# hyperparameters   
DATASET = args.dataset       
SEED = 20
EPOCH = args.epoch
LR = 1e-4
# BATCH_SIZE = 128
BATCH_SIZE = args.batch_size
P = args.p
MODEL_ARCH = args.model
R_RATIO = args.r_ratio

setup_seed(SEED)
# WIDTH_LIST = [16,32,48,64,15,17,18,19,20]

TRAIN_NUM = [200,400,600,800,1000,2000,4000,6000,8000,10000,20000,40000,50000]
DEVICE = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
res_path = '/home/zhiyu/codes/TensorComp/results/train_num/'

# data
# if DATASET == 'cifar10':
#     trainloader, testloader = load_cifar10_with_noise(noise_prob=P, batch_size=BATCH_SIZE)
# elif DATASET == 'cifar100':
#      trainloader, testloader = load_cifar100_with_noise(noise_prob=P, batch_size=BATCH_SIZE)

# model
for train_num in TRAIN_NUM:
     width = 16
     trainloader, testloader = load_cifar10_with_noise_train_num(noise_prob=P, batch_size=BATCH_SIZE, train_num=train_num)

     if DATASET == 'cifar10':
        num_classes = 10
     else:
         raise NotImplementedError
     
     if MODEL_ARCH == 'resnet18':
         model = make_resnet18k(k=width, num_classes=num_classes)
     elif MODEL_ARCH == 'tkresnet18':
         model = make_tkresnet18k(k=width, num_classes=num_classes,rank_ratio=R_RATIO)
     #     pdb.set_trace()
     
     model.to(DEVICE)
     train_error_lst, test_error_lst, epoch_lst = train_model(model, trainloader, testloader, epochs=EPOCH)
     res_dict = {'train_error': train_error_lst, 'test_error': test_error_lst, 'epoch': epoch_lst}
     res_dict = pd.DataFrame(res_dict)

     if MODEL_ARCH == 'resnet18':
         res_name = f'{train_num}_{DATASET}_w{width}_p{P}_batch{BATCH_SIZE}_epoch{EPOCH}.csv'
     elif MODEL_ARCH == 'tkresnet18':
         res_name = f'TK_R{R_RATIO}_{train_num}_{DATASET}_w{width}_p{P}_batch{BATCH_SIZE}_epoch{EPOCH}.csv'
     elif MODEL_ARCH == 'ttresnet18':
         res_name = f'TT_R{R_RATIO}_{train_num}_{DATASET}_w{width}_p{P}_batch{BATCH_SIZE}_epoch{EPOCH}.csv'
     res_dict.to_csv(os.path.join(res_path, res_name))

     # torch.save(model.state_dict(), f'./models/resnet18_cifar10_w{width}_p{P}.pth')
     # print(f'w={width} finished')