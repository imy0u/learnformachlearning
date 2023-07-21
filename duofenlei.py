'''prepare dataset
design model using class
construct loss'''
import torch
from torchvision import transforms#对数据进行处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


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
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)
        # 为什么降维分多次
    
    def forward(self,x):
        x=x.view(-1,784)
        '''
        view的用处
        '''
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        return self.l5(x)
    # 最后一层不做激活，不进行非线性变换

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