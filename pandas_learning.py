import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# se = pd.Series([1, 2, 3, np.nan, 1])
# dates = pd.date_range('20200101', periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

# 类型查看
# print(se.dtypes)
# print(df.dtypes)
# print(df.index)
# print(df.columns)
# print(df.values)

# 转置
# print(df.T)

# 排序输出
# print(df.sort_index(axis=1,ascending=False))
# print(df.sort_values(by='B'))

# 索引
# print(df)
# print(df['A'])
# print(df[3:])
# print(df[3:3]) # 空对象

# 用loc的方法
# print(df.loc['20200101'])
# print(df.loc[:,['A','B']])
# print(df.loc['20200102',['A','B']])

# 用iloc
# print(df.iloc[:,:])
# print(df.iloc[[1, 3, 5], 3])

# 筛选
# print(df[df.A > 8])

# 赋值
# df.iloc[2, 2] = 1111
# df.loc['20200101', :] = 20
# df.B[df.A > 4] = 0
# df['F'] = np.nan
# df['E'] = pd.Series(np.arange(1, 7), index=dates)
# print(df)

# 处理丢失数据
# df.iloc[0, 1] = np.nan
# df.iloc[1, 2] = np.nan
# # print(df.dropna(axis=0, how='any'))  # df本身的值并不会改变
# # print(df.dropna(axis=1, how='all'))  # 某一列全0才会丢弃
# # print(df.fillna(value=0))
# # print(df.isnull())
# print(np.any(df.isnull()))

# 合并
# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
# df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
# df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
# s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# res = pd.concat([df1, df2, df3], ignore_index=True)
# print(res)

# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
# df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
# res = pd.concat([df1, df2], join='outer', ignore_index=True, axis=0)
# print(res)

# res = df1.append([df2, df3], ignore_index=True)
# res = df1.append(s, ignore_index=True)
# print(res)

# merge合并
# left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
# right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
# print(left)
# print(right)
# res = pd.merge(left, right, on='key')
# print(res)
# left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K1', 'K0', 'K1'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
# right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
#                       'key2': ['K0', 'K0', 'K0', 'K0'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
# res = pd.merge(left, right, on=['key1', 'key2'], how='inner')  # how有inner、outer、left、right4种可传参数
# print(res)

# indicator显示合并方式
# df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
# df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
# res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
# print(res)

# 依照index合并
# left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
#                      'B': ['B0', 'B1', 'B2']},
#                     index=['K0', 'K1', 'K2'])
# right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
#                       'D': ['D0', 'D2', 'D3']},
#                      index=['K0', 'K2', 'K3'])
# res = pd.merge(left, right, left_index=True, right_index=True, how='outer')  # how也有4种可选参数
# print(res)

# 解决overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='outer')
print(res)
