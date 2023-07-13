'''classification'''
'''
the minist dataset
'''
import torchvision
train_set=torchvision.datasets.CIFAR10(root='./dataset/mnist',train=True,download=True)
test_set=torchvision.datasets.CIFAR10(root='./dataset/mnist',train=False,download=True)