'''
softmax线性回归
'''
'''
数据迭代器
是个接口，可以遍历集合的对象
为各种容器提供公共接口

'''
import torch
from IPython import display
from d2l import torch as d2l

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
"""
    Download the Fashion-MNIST dataset and then load it into memory.
    这里我将    ../data
    修改为      C:/Users/cas/Desktop/learnforpython/data/
"""

# 初始化模型参数
'''
输入：这里将图像看作长度为784的向量，目前暂时
只把每个像素位置看作一个特征 1*784
输出：10维度

权重：784*10
偏执：1*10
使用公式：y^= softmax(y)
y=x*w+b   1*10= 1*784  * 784*10  + 1*10
'''
num_inputs=784
num_outputs=10
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_inputs,requires_grad=True)

# 定义softmax函数
def softmax(x):
    x_exp=torch.exp(x)
    partition=x_exp.sum(1,keepdim=True)
    # sum（axis，keepdim=true）  axis=1表示可以理解为按照列的顺序，对行进行操作
    #keepdim表示保持原数组的维度
    return x_exp/partition

#定义模型
def net(x):
    return softmax(torch.matmul(x.reshape((-1,w.shape[0])),w)+b)
'''
搞明白 reshape和shape
reshape函数表示不改变数据的情况下为数组赋予新的形状
numpy.reshape(a, newshape)
'''

# 定义损失函数
# 交叉熵损失函数，最常用的损失函数，有点不理解
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    '''
    将预测出来的y值和真实的y值进行比较
    '''
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        '''
        y_hat是矩阵，第二个维度存储每个类的预测分数
        '''
        y_hat=y_hat.argmax(axis=1)
        #用argmax获得每行中最大的元素的索引来获得预测分数
        # axis=0按照行方向
        # axis=1按照列方向
    cmp=y_hat.type(y.dtype)==y
    #比较两个矩阵同位置是否相等，相等置为1
    return float(cmp.type(y.dtype).sum())
# 求和，并返回值
'''
对于任意数据迭代器data——iter 可以访问的数据集，我们可以
评估在任意模型net精度
'''
class Accmulator:
    '''变量累加'''
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b)for a,b in zip(self.data,args)] 
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
def evaluate_accuracy(net,data_iter):
    '''计算在指定数据集桑模型的精度'''
    if isinstance(net,torch.nn.Module):
        # isinstance可以判断一个变量的类型
        net.eval()#将模型设置为评估模式
    metric=Accmulator(2)
    with torch.no_grad():
        for x,y in data_iter:
            metric.add(accuracy(net(x),y),y.numel())
    return metric[0]/metric[1]
