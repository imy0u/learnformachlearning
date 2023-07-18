'''dataloader'''
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len 
    
dataset=DiabetesDataset('..\data\diabetes.csv.gz')
train_loader=DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2)
class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.ReLU()
    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x
    
model=Model()
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        y_pred=model(inputs)
        loss=criterion(y_pred,labels)
        print(epoch,i,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        