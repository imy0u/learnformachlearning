'''
多层感知机
'''
import torch
from d2l import torch as d2l
import matplotlib
from torch import nn

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

'''
激活函数
'''
def relu(x):
    a=torch.zeros_like(x)
    return torch.max(x,a)

'''
模型
'''
def net(x):
    x=x.reshape((-1,num_inputs))
    h=relu(x@W1+b1)
    return(h@W2+b2)
'''
损失函数
'''
loss=nn.CrossEntropyLoss(reduction='none')
'''
训练
'''
num_epochs,lr=10,0.1
# 迭代周期10   学习率0.1
updater=torch.optim.SGD(params,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)