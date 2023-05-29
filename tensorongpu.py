'''
tensor on gpu
用方法to（）可以将tensor在cpu和gpu之间相互移动
'''
import torch

if torch.cuda.is_available():
    device=torch.device("cuda")
    y=torch.ones_like(x,device=device)
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))#同时还可以改数据类型