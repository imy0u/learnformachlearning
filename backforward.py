'''
反向传播
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=torch.tensor([1.0])
w.requires_grad=True
w_list=[]
epoch_list=[]
w_list=[]
epoch_list=[]
def forward(x):
    return x*w
def cost(xs,ys):
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost+=(y_pred-y)**2
    return cost/len(xs)
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)

print("predice (before training)",4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w.grad.item())
        w.data=w.data-0.01*w.grad.data
        w.grad.data.zero_()
    w_list.append(w.data.item())
    epoch_list.append(epoch)   
    print("progress:",epoch,l)

print("predict (after training)",4,forward(4).item())

plt.plot(epoch_list,w_list)
plt.ylabel('w')
plt.xlabel('epoch')
plt.show()