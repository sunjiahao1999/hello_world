import torch
from torch.autograd import Variable
import numpy as np

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tesnor2array = torch_data.numpy()
#
# print(np_data, '\n', torch_data, '\n', tesnor2array)

data = np.array([[-1, -5], [1, 2]])
tensor = torch.FloatTensor(data)
# print(np.sin(data))
# print(torch.sin(tensor))
#
# print(np.dot(data.T, data))  # 矩阵乘法使用matmul与dot都行
# print(torch.mm(tensor.T, tensor))  # 如果使用torch.mm必须两个都是矩阵，而非向量,torch.dot不能用于矩阵相乘,得到的结果必须是一个向量

# variable = Variable(tensor, requires_grad=True)  # variable是一种节点类型，可以进行梯度反向传播，而tensor不行
# t_out = torch.mean(tensor * tensor)
# v_out = torch.mean(variable * variable)
#
# print(t_out)
# print(v_out)
#
# v_out.backward()
# print(variable.grad)  # 梯度反向传播
# print(variable)
# print(variable.data)  # 又变回tensor的形式


