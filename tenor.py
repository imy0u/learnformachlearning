'''
tensor和numpy相互转换
很容易使用numpy（）和from_numpy将tensor和numpy中的数组相互转换
这两个函数所产生的tensor和numpy中的数组共享相同的内存
改变其中一个时另一个也会改变
'''
#使用numpy（）将tensor转换成numpy数组
import torch

a=torch.ones(5)
b=a.numpy()
print(a,b)

a+=1
print(a,b)
b+=1
print(a,b)
#numpy数组转tensor
import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
print(a,b)

a+=1
print(a,b)
b+=1
print(a,b)