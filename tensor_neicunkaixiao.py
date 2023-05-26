'''
运算的内存开销
索引和view是不会开辟新内存的，而像运算是会开新的内存
'''
import torch
x=torch.tensor([1,2])
y=torch.tensor([3,4])
id_before=id(y)
# y=y+x
# print(id(y)==id_before)#false
'''
如果向指定到原来的y的内存，我们可以使用前面介绍的索引来进行替换操作
'''
y[:]=y+x
print(id(y)==id_before)#通过【：】写进y的内存里
'''
还可以使用运算符全名函数中的out参数
或
自加运算发+=达到上述效果'''
id_before=id(y)
torch.add(x,y,out=y)
# y+=x
#y.add(x)
print(id(y)==id_before)