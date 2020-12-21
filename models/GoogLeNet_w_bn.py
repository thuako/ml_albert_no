import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()        

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, padding=3)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=5)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)   
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, **kwargs),
                            nn.BatchNorm2d(out_channels),   #Batch norm here
                            nn.ReLU()
                            )
    def forward(self, x):
        return self.conv(x)

class BasicConv2d_v1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, **kwargs):
        super(BasicConv2d_v1, self).__init__()
        self.conv = nn.Sequential(
                            nn.Dropout2d(dropout),
                            nn.Conv2d(in_channels, out_channels, **kwargs),
                            nn.BatchNorm2d(out_channels),   #Batch norm here
                            nn.ReLU()
                            )
    def forward(self, x):
        return self.conv(x)

class Inception_v1(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception_v1, self).__init__()
        
        self.branch1 = BasicConv2d_v1(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d_v1(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d_v1(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d_v1(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d_v1(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d_v1(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)   
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)        


class GoogLeNet_v1(nn.Module):

    def __init__(self):
        super(GoogLeNet_v1, self).__init__()        

        self.conv1 = BasicConv2d_v1(3, 64, kernel_size=7, padding=3)
        self.conv2 = BasicConv2d_v1(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d_v1(64, 192, kernel_size=5)

        self.inception3a = Inception_v1(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_v1(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception_v1(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_v1(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_v1(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_v1(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_v1(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception_v1(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_v1(832, 384, 192, 384, 48, 128, 128)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x        



class GoogLeNet_v2(nn.Module):

    def __init__(self):
        super(GoogLeNet_v2, self).__init__()        

        self.conv1 = BasicConv2d_v1(3, 64, kernel_size=7, padding=3)
        self.conv2 = BasicConv2d_v1(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d_v1(64, 192, kernel_size=5)

        self.inception3a_shortcut = nn.Conv2d(192, 256, kernel_size=1)
        self.inception3a = Inception_v1(192, 64, 96, 128, 16, 32, 32)
        
        self.inception3b_shortcut = nn.Conv2d(256, 480, kernel_size=1)
        self.inception3b = Inception_v1(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a_shortcut = nn.Conv2d(480, 512, kernel_size=1)
        self.inception4a = Inception_v1(480, 192, 96, 208, 16, 48, 64)

        self.inception4b_shortcut = nn.Sequential()
        self.inception4b = Inception_v1(512, 160, 112, 224, 24, 64, 64)

        self.inception4c_shortcut = nn.Sequential()
        self.inception4c = Inception_v1(512, 128, 128, 256, 24, 64, 64)

        self.inception4d_shortcut = nn.Conv2d(512, 528, kernel_size=1)
        self.inception4d = Inception_v1(512, 112, 144, 288, 32, 64, 64)

        self.inception4e_shortcut = nn.Conv2d(528, 832, kernel_size=1)
        self.inception4e = Inception_v1(528, 256, 160, 320, 32, 128, 128)


        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a_shortcut = nn.Sequential()
        self.inception5a = Inception_v1(832, 256, 160, 320, 32, 128, 128)

        self.inception5b_shortcut = nn.Conv2d(832, 1024, kernel_size=1)
        self.inception5b = Inception_v1(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)

        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        
        x = self.relu(self.inception3a(x) + self.inception3a_shortcut(x) )
        x = self.relu(self.inception3b(x) + self.inception3b_shortcut(x) )
        x = self.maxpool3(x)

        x = self.relu(self.inception4a(x) + self.inception4a_shortcut(x) )
        x = self.relu(self.inception4b(x) + self.inception4b_shortcut(x) )
        x = self.relu(self.inception4c(x) + self.inception4c_shortcut(x) )
        x = self.relu(self.inception4d(x) + self.inception4d_shortcut(x) )
        x = self.relu(self.inception4e(x) + self.inception4e_shortcut(x) )
        x = self.maxpool4(x)

        x = self.relu(self.inception5a(x) + self.inception5a_shortcut(x) )
        x = self.relu(self.inception5b(x) + self.inception5b_shortcut(x) )
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x        




class GoogLeNet_v3(nn.Module):

    def __init__(self):
        super(GoogLeNet_v3, self).__init__()        

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, padding=3)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=5)

        self.inception3a_shortcut = nn.Conv2d(192, 256, kernel_size=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        
        self.inception3b_shortcut = nn.Conv2d(256, 480, kernel_size=1)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a_shortcut = nn.Conv2d(480, 512, kernel_size=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)

        self.inception4b_shortcut = nn.Sequential()
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)

        self.inception4c_shortcut = nn.Sequential()
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)

        self.inception4d_shortcut = nn.Conv2d(512, 528, kernel_size=1)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)

        self.inception4e_shortcut = nn.Conv2d(528, 832, kernel_size=1)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)


        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a_shortcut = nn.Sequential()
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)

        self.inception5b_shortcut = nn.Conv2d(832, 1024, kernel_size=1)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)

        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        
        x = self.relu(self.inception3a(x) + self.inception3a_shortcut(x) )
        x = self.relu(self.inception3b(x) + self.inception3b_shortcut(x) )
        x = self.maxpool3(x)

        x = self.relu(self.inception4a(x) + self.inception4a_shortcut(x) )
        x = self.relu(self.inception4b(x) + self.inception4b_shortcut(x) )
        x = self.relu(self.inception4c(x) + self.inception4c_shortcut(x) )
        x = self.relu(self.inception4d(x) + self.inception4d_shortcut(x) )
        x = self.relu(self.inception4e(x) + self.inception4e_shortcut(x) )
        x = self.maxpool4(x)

        x = self.relu(self.inception5a(x) + self.inception5a_shortcut(x) )
        x = self.relu(self.inception5b(x) + self.inception5b_shortcut(x) )
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x        


        