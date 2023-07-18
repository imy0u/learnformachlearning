'''manual data feed'''
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
dataset=DiabetesDataset()
train_Loader=DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2)

model=Model()
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
xy=np.loadtxt('./data/diabetes.csv.gz',delimiter=','dtype=np.float(32))
# 到处一组minibatch供我们快速使用
x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()