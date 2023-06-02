'''
线性回归
输出是一个连续值，也实用于回归问题
线性回归都是单层神经网络
'''
'''
线性回归的从零开始实现
'''
import torch
from IPython.display import display
import matplotlib_inline
from matplotlib import pyplot as plt
import numpy as np
import random
'''
构造人工数据集
'''
# num_inputs=2
# num_examples=1000
# true_w=[2,-3.4]
# true_b=4.2
# features=torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
# #features是每一行是一个长度为2的向量
# labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# #labels是每一行是一个长度为1的向量（标量）
# labels+=torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))
# # print(features[0],labels[0])

def synthetic_data(w,b,num_examples):
    '''y=Xw+b+噪音.'''
    X=torch.normal(0,1,(num_examples,len(w)))#均值为0，方差为1的随机数
    y=torch.matmul(X,w)+b#Xw+b
    y+=torch.normal(0,0.01,y.shape)#均值为0，方差为0.01，形状和y一样
    return X,y.reshape((-1,1))#reshape(-1,1)转化为1行
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,10000000)

def use_svg_display():
    # display.set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#这里不明白

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize

def showshowkan():#展示
    set_figsize()
    plt.scatter(features[:,1].numpy(),labels.numpy(),1)
    plt.show()

def data_iter(batch_size,features,labels):#每次输出batch_size个
    num_examples=len(features)
    indices=list(range(num_examples))#生成样本长度个数列
    random.shuffle(indices)#将样本顺序随机打乱
    for i in range(0,num_examples,batch_size):#每次跳batch_size个大小
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
#yield是return的同胞兄弟
# yield返回yield的函数则返回一个可迭代的 generator（生成器
# ）对象，你可以使用for循环或者调用next(
# )方法遍历生成器对象来提取结果。
def read_shuju():
    for X,y in data_iter(batch_size,features,labels):
        print(X,y)
        break
batch_size=10
'''初始化模型参数'''
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
'''定义模型'''
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
'''定义损失函数'''
def squared_loss(y_hat,y):
    '''
    均方误差
    没有算均值
    '''
    return(y_hat-y.reshape(y_hat.shape))**2/2
'''定义优化算法'''
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():#这里表示不自动求导，自动设置为false
        for param in params:
            param -= lr * param.grad / batch_size#步伐*
            param.grad.zero_()
            '''
            斜率和梯度的区别
            梯度是一个方向向量，表示函数在该点沿着这个方向变化最快
            '''
'''训练'''
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')