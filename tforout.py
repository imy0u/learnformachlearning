'''
tidu
因为out是一个标量，所以调用backward时不需要指定求导变量
'''
import torch
out.backward()

print(x.grad)