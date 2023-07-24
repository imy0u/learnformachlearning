'''
cnn
'''
'''convolutional layer'''
import torch
in_channels,out_channels=5,10
width,height=100,100
kernel_size=3
batch_size=1#批量的维度

input=torch.randn(batch_size,
                  in_channels,
                  width,
                  height)
conv_layer=torch.nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size)
output=conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
'''
run_out
torch.Size([1, 5, 100, 100])
torch.Size([1, 10, 98, 98])
torch.Size([10, 5, 3, 3])
输入通道数，输入通道数，长，高
'''