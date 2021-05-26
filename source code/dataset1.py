import pandas as pd
import numpy as np
from random import randrange,seed
import matplotlib.pyplot as plt

# 以下函数用来划分数据集，可以选择分层划分或者随机划分
def cross_validation_split(data: pd.DataFrame, folds = 3, Target_Ratio = False, Random_State = 0):
    if Target_Ratio:
        group = data.groupby(data.columns[-1])
        dataset_split = []
        for _ in range(folds):  # 在 dataset_split 中添加[]
            dataset_split.append([])
        for tmp in range(folds):
            for indx in group.size().index:   # groupby的值(list)
                for _ in cross_validation_split(group.get_group(indx),folds,Random_State)[tmp]:
                    dataset_split[tmp].append(_)
        return dataset_split
    else:
        dataset = data.values.tolist()
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / folds)
        for i in range(folds):
            fold = list()
            while len(fold) < fold_size:
                seed(Random_State)
                index = randrange(len(dataset_copy))  # 随机生成下标
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split



plt.figure(figsize=(10, 5))
x = np.arange(0, 4*np.pi, 0.25)    # 等距从0到4Π取数据点，跨度为0.25.共51个数据点
y = np.sin(x) + 0.25 * np.random.rand(51) - 0.25*np.random.rand(51)  # 在sinx函数的基础上，生成51个1以下的随机实数，且加上正负权重0.25.使得在目标点上下抖动
plt.plot(x, np.sin(x), label='curve')  # 画出该函数的曲线图
plt.scatter(x, y, color='red', label='data point')  # 画出取到的数据点
plt.legend()  # 图例在默认位置
plt.ylabel('value')  # y轴名字
plt.xlabel('x')   # X轴名字
plt.savefig('./datapoint.svg', format='svg')  # 保存为矢量图

data1 = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis =1)  # 数据拼接
data1.to_csv('data1.csv') # 数据写回
