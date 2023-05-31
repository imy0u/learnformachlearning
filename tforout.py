'''
tidu
因为out是一个标量，所以调用backward时不需要指定求导变量
'''
import torch
# x=torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
# y=2*x
# z=y.view(2,2)
# print(z)
# v=torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
# z.backward(v)
# print(x.grad)
'''
x=torch.tensor(1.0,requires_grad=True)
y1=x**2
with torch.no_grad():
    y2=x**3#被no_grad包裹了，梯度不会被回传
y3=y1+y2

print(x.requires_grad)
print(y1,y1.requires_grad)
print(y2,y2.requires_grad)
print(y3,y3.requires_grad)
y3.backward()
print(x.grad)
'''

x=torch.ones(1,requires_grad=True)

print(x.data)
print(x.data.requires_grad)

y=2*x
x.data*=100

y.backward()
print(x)#更改data的值会影响tensor的值，但是不会被记录在计算图中，不影响梯度传播

print(x.grad)