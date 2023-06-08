'''
softmax回归
虽然叫做回归，但其实是分类算法
'''
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
from IPython import display

'''下载数据集'''
d2l.use_svg_display()
trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="C:/Users/cas/Desktop/learnforpython/data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="C:/Users/cas/Desktop/learnforpython/data",train=False,transform=trans,download=True)
len(mnist_train),len(mnist_test)
'''可视化数据集的函数'''
def get_fashion_mnist_labels(labels):
    '''返回数据集的文本标签'''
    text_labels=[
        't-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaer','bag','ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    '''plot a list of images.'''
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    # plt.show()
    return axes
X,y=next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))

'''读取小批量数据，大小为batch_size'''
batch_size=256

def get_dataloader_workers():
    '''使用四个进程来读取数据
    '''
    return 4

train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

timer=d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():2f} sec')

