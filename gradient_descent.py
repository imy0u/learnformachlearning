'''
梯度下降算法
贪心算法，每次都往梯度的负方向走，-梯度乘上学习率
会出现局部最优解，而非全局最优解
在机器学习中，局部最优点比较少
容易出现鞍点，梯度为零的一段连续点，会导致梯度下降算法无法进行下去
在三维空间中出现类似马鞍的一个平面，会同时出现在不同切面中出现最大点和最小点
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=3
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

def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)
print('predict (before training)',4,forward(4))
for epoch in range(1000):
    cost_val=cost(x_data,y_data)
    grad_val=gradient(x_data,y_data)
    w-=0.0015*grad_val
    print('epoch:',epoch,'w=',w,'loss=',cost_val)
    w_list.append(cost_val)
    epoch_list.append(epoch)
print('predict (after training)',4,forward(4))

# 绘制二维曲线图
plt.plot(epoch_list,w_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

'''
随机梯度下降算法
体现在'''