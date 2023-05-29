import torch
x=torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)
y=x+2
print(y)
print(y.grad_fn)
'''tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
<AddBackward0 object at 0x0000024C120DCF40>
因为grad——fn返回的这个addbackward，表明进行了加法运算
'''
print(x.is_leaf,y.is_leaf)#x是直接创建的，也被称作为叶子节点
z=y*y*3
out=z.mean()
print(z,out)

a=torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)