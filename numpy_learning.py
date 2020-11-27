import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)
# b = np.zeros((3, 4), dtype=int)
# c = np.arange(10).reshape((2,5))
# d=np.linspace(1,10,5)
# print(a.ndim)
# print(a.shape)
# print(a.dtype)
# print(b)
# print(c)
# print(d)
# print(a**2)
# print(a+a)
# print(np.sin(a))
# print(a<3)

# 矩阵乘法
# print(np.dot(a, a))
# print(a.dot(a))
# print(np.matmul(a, a.T))

# e = np.random.random((2, 4))  # 两个括号,随机生成0至1的数组成2行4列的矩阵
# e = np.random.rand(2,4)
# f = np.random.randn(3，3)  # 生成的数据服从标准正态分布
# print(e)
# print(f)

# print(np.sum(a))
# print(np.max(a,axis=1))
# print(np.min(a,axis=0))
# print(np.argmin(a))
# print(np.argmax(a))

# # 平均值、中位数
# print(np.average(a))
# print(a.mean())
# print(np.median(a,axis=0))
# print(np.cumsum(a))  # 逐步累加
# print(np.diff(a))  # 逐步累差
# print(np.nonzero(a)) #找非零元素,分别输出每个非零元素的行与列组成的矩阵
# print(np.sort(a))  # 逐行升序排列

# # 转置
# print(a.T)
# print(np.transpose(a))
# print((a.T).dot(a))
# print(np.clip(a,5,7)) #将大于7小于5的数转化为7和5

# # 索引
# print(a[1, 2])
# print(a[1][2])
# print(a[:, 1])  # 第2列
# print(a[1, 1:3])  # 第2行的2 3列

# # 迭代每一行
# for x in a:
#     print(x)
#
# # 迭代每一列
# for x in a.T:
#     print(x)

# print(a.flat)  # 返回一个numpy独有的迭代器
# print(a.flatten())  # 将矩阵展开输出
# for i in a.flat:
#     print(i)

# 横向合并和纵向合并
# print(np.hstack((a, a)))  # 注意两个括号
# print(np.vstack((a, a)))  # horizonal和vertical
# print(np.concatenate((a, a, a), axis=1))

# # 向量的转置
# b = np.array([1, 2, 3, 4])
# print(b.shape)  # 显示的是向量的维度
# print(b[:, np.newaxis])  # 不能直接通过b.T来转置

# 矩阵分割
# print(np.split(a, 3, axis=0))  # 再用vstack可复原
# print(np.array_split(a, 2, axis=1))  # 不等的分割
# print(np.hsplit(a, 3))

# 深复制
# b=a.copy()
