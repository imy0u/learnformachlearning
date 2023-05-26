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
'''
操作
tensor的各种操作
'''
'''算术操作'''
y= torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
y.add(x)
print(y)

'''索引操作，我们还可以使用索引操作访问，索引出来的结果与元数据共享内存，
修改一个，另一个也会跟着修改
'''
y=x[2,:]
y+=1
print(y)
'''
改变形状
用view来改变tensor的形状

'''
y=x.view(15)#view仅仅是改变了对这个张量的观察角度
z=x.view(-1,5)
print(x.size(),y.size(),z.size())
print(x,y,z)
'''如果向返回一个真正新的副本（不共享内存）
reshape（）可以改变形状，但是这个函数并不能保证返回的是其拷贝
推荐使用clone创造一个副本然后再使用view'''
x_cp=x.clone().view(15)
x-=1
print('___________________________________________')
print(x)
print(x_cp)

x=torch.randn(1)
print(x)
print(x.item())#将tensor转换程一个python number
'''
线性代数
'''