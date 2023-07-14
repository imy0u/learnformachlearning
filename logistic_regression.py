'''
logistic regression
'''
import torch
x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[0],[0],[1]])

class Logisticregessionmodel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Logisticregessionmodel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
    
model=Logisticregessionmodel()

criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()