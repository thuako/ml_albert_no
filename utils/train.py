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
import time


def train(hyper_param_dict, model, device):    

    transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=hyper_param_dict['data root'],
                                    train=True, 
                                    transform=transform,
                                    download=True)

    test_dataset = datasets.CIFAR10(root=hyper_param_dict['data root'],
                                    train=False, 
                                    transform=transforms.ToTensor())

    loss_function = torch.nn.CrossEntropyLoss()
    if hyper_param_dict['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_param_dict['lr'], betas=(hyper_param_dict['beta1'], hyper_param_dict['beta2']), 
                                weight_decay=hyper_param_dict['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyper_param_dict['lr'], momentum=hyper_param_dict['momentum'], 
                                weight_decay=hyper_param_dict['weight_decay'])
    print(optimizer, model)

    model.train()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyper_param_dict['batch'], shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=hyper_param_dict['batch'], shuffle=False, num_workers=2)

    train_acc_list, test_acc_list, train_loss_list, test_loss_list = [], [], [], []

    
    start = time.time()
    for epoch in range(hyper_param_dict['epochs']) :
            # start training
            model.train()
            train_loss, train_acc, total, correct = 0, 0, 0, 0

            train_start = time.time()
            for i, (images, labels) in enumerate(train_loader) :
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    output = model(images)
                    train_loss = loss_function(output, labels)
                    train_loss.backward()
                    optimizer.step()

                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                    total += labels.size(0)
            train_end = time.time()

            print ("Epoch [{}] Loss: {:.4f}  Epoch time : {:.4f}".format(epoch+1, train_loss.item(), train_end - train_start))
            #save train result
            train_loss_list.append(train_loss / total)
            train_acc_list.append(correct / total * 100.)

            # start evaluation
            model.eval()    
            test_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                    for images, labels in test_loader :
                            images, labels = images.to(device), labels.to(device)

                            output = model(images)
                            test_loss += loss_function(output, labels).item()

                            pred = output.max(1, keepdim=True)[1]
                            correct += pred.eq(labels.view_as(pred)).sum().item()

                            total += labels.size(0)

            print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                    test_loss /total, correct, total,
                    100. * correct / total))
            #save test result
            train_loss_list.append(train_loss / total)
            train_acc_list.append(correct / total * 100.)

    end = time.time()
    print("Time ellapsed in training is: {}".format(end - start))

    hyper_param_dict['training time'] = end - start

    '''
    Step 5
    '''
    # save file by numpy
    np_train_acc_list, np_test_acc_list = np.array(train_acc_list), np.array(test_acc_list)
    np_train_loss_list, np_test_loss_list = np.array(train_loss_list), np.array(test_loss_list)

    if not os.path.isdir('./Result'):
            os.makedirs('./Result')

    # Make save directory

    project_name = hyper_param_dict['project']

    now = time.localtime()
    save_time = str(now.tm_mday) + '_' + str(now.tm_hour) + '_' + str(now.tm_min)
    if project_name is not None:
        save_dir = os.path.join(os.getcwd(), 'Result', project_name ,save_time)
    else:
        save_dir = os.path.join(os.getcwd(), 'Result', save_time)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print(f'save dir : {save_dir}')
    np.save(save_dir + '/np_train_acc_list', np_train_acc_list)
    np.save(save_dir + '/np_train_loss_list', np_train_loss_list)
    np.save(save_dir + '/np_test_acc_list', np_test_acc_list)
    np.save(save_dir + '/np_test_loss_list', np_test_loss_list)

    with open( save_dir + '/hyper.pickle','wb') as fw:
        pickle.dump(hyper_param_dict, fw)
