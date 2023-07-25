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

batch_size=64
transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#均值 标准差 这两个数字来历 是计算出来的，是站在巨人肩膀上的
])
train_dataset=datasets.MNIST(root='./data/minst/',
                             train=True,
                             download=True,
                             transform=transforms)
train_loader=DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=batch_size)
test_dataset=datasets.MNIST(root='./data/minst/',
                            train=True,
                            download=True,
                            transform=transforms)
test_loader=DataLoader(test_dataset,
                       shuffle=False,
                       batch_size=batch_size)
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
        self.fc=nn.Linear(1408,10)#全连接 1408根据输出size来求  求得时候，先跳过这段程序
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

        '''可以先跳过下面两行，直接输出tensor.size'''
        x=x.view(in_size,-1)
        # view( )函数相当于numpy中的resize( )函数，都是用来重构(或者调整)张量维度的，用法稍有不同。
        # view(-1)将张量重构成了1维的张量。
        x=self.fc(x)
        return x
    

model=Net()

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
'''
optimizer是神经网络优化器 SGD是最普通的训练方法，即随机梯度下降算法
momentum
“冲量”这个概念源自于物理中的力学，表示力对时间的积累效应。
在普通的梯度下降法x+=v
中，每次x的更新量v为v=−dx∗lr，其中dx为目标函数func(x)对x的一阶导数，。
当使用冲量时，则把每次x的更新量v考虑为本次的梯度下降量−dx∗lr与上次x的更新量v乘上一个介于[0,1]的因子
momentum的和，即
v=−dx∗lr+v∗momemtum
想象成一个人在斜坡上，被重力作用下，往下滑
model.parameters()意义在于返回一个生成器，迭代器，存储weight
'''
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()
        outputs=model(inputs)
        # 调用魔法函数
        loss=criterion(outputs,target)
        loss.backward()
        # 反向传播，计算当前梯度
        optimizer.step()
        # 根据梯度更新网络参数
        running_loss+=loss.item()
      
        if batch_idx%300==299:
            print('[%d,%5d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            # 测试集合
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            '''torch.max使用讲解
            输入
                torch.max(input,dim)函数
                input 是softmax 函数输出的一个tensor
                dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            输出
                函数返回两个tensor，第一个tensor是每行的最大值，其中值最大为1
                第二个tensor是每行得最大值得索引
            '''
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('accuracy on test set:%d %%'%(100*correct/total))

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()