'''linear regression'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[2.0],[4.0],[6.0]])
epoch_list=[]
item_list=[]

class linearmodel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear=torch.nn.Linear(1,1)
        # 定义输入输出都是一维的

    def forward(self,x):
        y_pred=self.linear(x)
        # 调用wx+b模型，并输入x
        '''    
        def forward(self, input: Tensor) -> Tensor:
            return F.linear(input, self.weight, self.bias)
            '''
        return y_pred
    
model=linearmodel()

criterion=torch.nn.MSELoss(size_average=False)
# 这里设置求损失不求均值啥意思
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# 设置优化器 设置学习速率
for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    epoch_list.append(epoch)
    item_list.append(loss.item())
    optimizer.zero_grad()
    # 设置梯度清零
    loss.backward()
    optimizer.step()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)

plt.plot(epoch_list,item_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()