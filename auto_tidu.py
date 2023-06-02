'''
自动求梯度
'''
import torch

x=torch.arange(4.0)
print(x)
x.requires_grad_(True)
y=2*torch.dot(x,x)#x内积乘上2
y.backward()#求导
print(x.grad)
print(x.grad==4*x)
x.grad.zero_()#梯度置零,如果不值零，会导致自动累加
y=x*x
y.sum().backward()
print(x.grad)
'''
'''
x.grad.zero_()
y=x*x
u=y.detach()
# 这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播

z=u*x

z.sum().backward()
print(x.grad==u)
'''

控制流的语句
'''
def f(a):
    b=a*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c

a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()

print(a.grad==d/a)