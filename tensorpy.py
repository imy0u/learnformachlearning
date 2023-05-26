'''
数据操作
pytorch中，tensor是存储和变换数据的主要工具
创建tensor
tensor是张量，可以看作多维数组。标量是零维张量，向量是一维张量
矩阵是二维张量
'''
import torch
x=torch.empty(5,3)#创建未初始化的tensor
x=torch.rand(5,3)#创建随机初始化的tensor
x=torch.zeros(5,3,dtype=torch.long)#创建一个long型全0的tensor
x=torch.tensor([5.5,3])#自己定义一个
x=x.new_ones(5,3,dtype=torch.float64)

x=torch.rand_like(x,dtype=torch.float)
print(x.size())
print(x.shape)
print(x)