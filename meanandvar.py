#计算均值和方差
def mean(values):#均值
    return sum(values)/float(len(values))
#计算求方差
def variance(values,mean):
    return sum([(x-mean)**2 for x in values])

#开始计算均值和方差
dataset = [[1.2,1.1],[2.4,3.5],[4.1,3.2],[3.4,2.8],[5,5.4]]
x= [row[0] for row in dataset]
print(type(x))
y=[row[1] for row in dataset]
print(y)
mean_x,mean_y=mean(x),mean(y)
var_x,var_y=variance(x,mean_x),variance(y,mean_y)
print('x 统计特性：均值=%.3f 方差=%.3f'%(mean_x,var_x))
print('y 统计特性：均值=%.3f 方差=%.3f'%(mean_y,var_y))
#计算协方差
def covariance(x,mean_x,y,mean_y):
    covar=0.0
    for i in range(len(x)):
        covar +=(x[i]-mean_x)*(y[i]-mean_y)
    return covar
#mean_x,mean_y=mean(x),mean(y)
covar=covariance(x,mean_x,y,mean_y)
print('协方差=%.3f'%(covar))
#计算回归系数
w1=covar/var_x
w0=mean_y-w1*mean_x
print('回归系数分别为：w0=%.3f,w1=%.3f'%(w0,w1))
#构建回归模型
def simple_linear_regression(w0,w1,test):
    predict=list()
    for row in test:
        y_model=w1*row[0]+w0
        predict.append(y_model)
    return predict
