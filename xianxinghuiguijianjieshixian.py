'''
线性回归的简洁实现
'''
import torch
from IPython.display import display
import matplotlib_inline
from matplotlib import pyplot as plt
import numpy as np
import random
from torch.utils import data
#引入处理数据的模块data
from d2l import torch as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)
def load_array(data_arrays,batch_size,is_train=True):
    '''
    构造一个pytorch数据迭代器
    '''
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
#shuffle表示是否随机打乱顺序，如果是train表示需要

batch_size=10
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

'''使用框架的预定义好的层'''
from torch import nn
net=nn.Sequential(nn.Linear(2,1))#表示层输入维度是2 输出维度是1

'''初始化模型参数'''
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
'''计算均方误差'''
loss=nn.MSELoss()
'''实例化SGD实例'''
trainer=torch.optim.SGD(net.parameters(),lr=0.03)
# net.parameters表示net里所有参数

num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch {epoch+1},loss {1:f}')
