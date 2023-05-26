'''
广播机制
上一个文件主要试验了两个形状相同的tensor做按元素运算
当对两个形状不同的tensor按照元素运算是，可能会出发广播机制
先适当复制元素十这两个tensor形状相同后再按元素运算
'''
import torch
x=torch.arange(1,3).view(1,2)
#arange函数表示从1到3轮着来，默认跳度是1，当然了，还可以设置为0.5，写在第三个参数位置

print(x)
y=torch.arange(1,4).view(3,1)
print(y)
print(x+y)