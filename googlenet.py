'''
信息融合，经过原数据经过某种计算得来
'''
import torch
from torchvision import transforms#对数据进行处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        # in_channels指明输入通道数
        super(InceptionA,self).__init__()
        '''
        
        '''
        self.branch1x1=nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool=nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]
        return torch.cat(outputs,dim=1)
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=nn.MaxPool2d(2)#池化
        self.fc=nn.Linear(1408,10)#全连接
        '''
        池化与卷积的不同点：卷积操作的卷积核是有数据（权重）的，
        而池化直接计算池化窗口内的原始数据，
        这个计算过程可以是选择最大值、选择最小值或计算平均值，
        分别对应：最大池化、最小池化和平均池化。由于在实际使用中最大池化是应用最广泛的池化方法，

'''

    def forward(self,x):
        in_size=x.size(0)#查看tensor的维度
        x=F.relu(self.mp(self.conv1(x)))#relu激活函数 10
        '''
        把“激活的神经元的特征”通过函数把特征保留并映射出来，即负责将神经元的输入映射到输出端。
        '''
        x=self.incep1(x)#88
        x=F.relu(self.mp(self.conv2(x)))#20
        x=self.incep2(x)#88
        x=x.view(in_size,-1)
        # view( )函数相当于numpy中的resize( )函数，都是用来重构(或者调整)张量维度的，用法稍有不同。
        # view(-1)将张量重构成了1维的张量。
        x=self.fc(x)
        return x