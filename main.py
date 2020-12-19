import torch
import torch.nn as nn
import math
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import datetime
import pickle
import os
import matplotlib.pyplot as plt


import utils
import models



#hyper parameter 

hyper_param_dict = { 
                'project' : 'VGG13.v1',
                'data root' : './Datasets/cifar10',
                'epochs' : 2,
                'batch' : 256, 
                'lr' : 0.05,
                'optimizer': 'SGD',
                'momentum' : 0.9,   
                'beta1' : 0.9,
                'beta2' : 0.999, 
                'weight_decay' : 5e-4
                }


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.VGG13()


#start training

utils.train(hyper_param_dict, model, device)



#find save dir path
if hyper_param_dict[ 'project'] is not None:
    save_dir = './Result/' + hyper_param_dict[ 'project'] + '/'
else:
    save_dir = './Result/'
dir_list = os.listdir(save_dir)

for dir_name in dir_list:
    if '_' in dir_name:
        load_dir = save_dir + dir_name
        train_acc = np.load(load_dir + '/np_train_acc_list.npy', allow_pickle=True)
        test_acc = np.load(load_dir + '/np_test_acc_list.npy', allow_pickle=True)
        train_loss = np.load(load_dir + '/np_train_loss_list.npy', allow_pickle=True)
        test_loss = np.load(load_dir + '/np_test_loss_list.npy', allow_pickle=True)

        with open( load_dir + '/hyper.pickle', 'rb') as fr:
            hyper_param_load = pickle.load(fr)

        plt.plot(np.arange(0, train_acc.shape[0]), train_acc, label=  'lr = ' + str(hyper_param_load['lr']))
    plt.legend()
    plt.title(hyper_param_dict['project'])
    #plt.show()