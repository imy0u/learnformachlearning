'''prepare dataset
design model using class
construct loss'''
import torch
from torchvision import transforms#对数据进行处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


batch_size=64
transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#均值 标准差 这两个数字来历 是计算出来的，是站在巨人肩膀上的
])

train_dataset=datasets.MNIST(root='./data/minst/',
                             train=True,
                             download=True,
                             transform=transforms)
train_loader=DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=batch_size)
test_dataset=datasets.MNIST(root='./data/minst/',
                            train=True,
                            download=True,
                            transform=transforms)
test_loader=DataLoader(test_dataset,
                       shuffle=False,
                       batch_size=batch_size)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)
    
    def forward(self,x):
        x=x.view(-1,784)
        '''
        view的用处
        '''
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        return self.l5(x)
    # 最后一层不做激活，不进行非线性变换

model=Net()

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
'''
什么用处
'''
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%5d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('accuracy on test set:%d %%'%(100*correct/total))

if __name__=='__main__':
    for epoch in range(15):
        train(epoch)
        test()