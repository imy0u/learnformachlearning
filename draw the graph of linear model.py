'''draw the graph of 线性模型 linear model
假设模型，
控制w的大小，使得mse不断变化
学习目的，线性模型的了解 以及训练的时候的可视化
最好的是训练的过程中就可以实现可视化
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
# exercise try to use the model in right-side,and draw the cost graph
# y=wx+b
# draw 3d graph
def forward(x):
    return x*w+b

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

w_list=[]
b_list=[]
mse_list=[]
for b in np.arange(0.0,4.0,0.1):
    for w in np.arange(0.0,4.1,0.1):
        print('w=',w)
        print('b=',b)
        l_sum=0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val=forward(x_val)
            loss_val=loss(x_val,y_val)
            l_sum+=loss_val
            print('\t',x_val,y_val,y_pred_val,loss_val)
        print('MSE=',l_sum/3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/3)
# 绘制二维曲线图
# plt.plot(w_list,mse_list)
# plt.ylabel('loss')
# plt.xlabel('w')
# plt.show()

#绘制散点图
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(xs=w_list,ys=b_list,zs=mse_list,s=30,c="g",depthshade=True,cmap="jet")
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ax3d = mp.gca(projection="3d")    # 同样可以实现

# ax.set_xlabel("w")
# ax.set_ylabel("b")
# ax.set_zlabel("mse")
ax.plot_trisurf(w_list,b_list,mse_list,cmap="jet")

plt.show()