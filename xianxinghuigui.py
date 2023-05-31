'''
线性回归
输出是一个连续值，也实用于回归问题
线性回归都是单层神经网络
'''
'''
线性回归的从零开始实现
'''
import torch
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import random
'''
构造人工数据集
'''
num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
features=torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
#features是每一行是一个长度为2的向量
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
#labels是每一行是一个长度为1的向量（标量）
labels+=torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))
# print(features[0],labels[0])

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize

set_figsize()
plt.scatter(features[:,1].numpy,labels.numpy(),1)