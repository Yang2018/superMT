import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
from sklearn.preprocessing import PolynomialFeatures
from random import randrange, seed
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

def takey(ll):
    # 用于从分割后的数据中分离出标签与属性
    X = []
    y = []
    for tmp in ll:
        y.append(tmp[-1])
        X.append(tmp[:-1])
    return np.array(X), np.array(y)  # 前一元素是属性，后一元素是标签

# 读入数据集
x = np.arange(0, 4*np.pi, 0.25)    # 等距从0到4Π取数据点，跨度为0.25.共51个数据点
y = np.sin(x) + 0.25 * np.random.rand(51) - 0.25*np.random.rand(51)  # 在sinx函数的基础上，生成51个1以下的随机实数，且加上正负权重0.25.使得在目标点上下抖动
data1 = pd.read_csv('../datasets/data1.csv', index_col = 0)


# 数据预处理：
newx = []
copyx = x
for i in x:
    newx.append([i])

folds = 5
X_train = []
maes = []   # 存储MAE值
r2s = []    # 存储r2score值
rmses = []  # 存储RMSE值

newdata1 = cross_validation_split(data1, folds=5) # 5则划分数据集

X_test = newdata1[0]
copy = [1, 2, 3, 4]
for y in copy:
    for z in newdata1[y]:
        X_train.append(z)

x_test, y_test = takey(X_test)
x_train, y_train = takey(X_train)
newy_test = []
for x in y_test:
    newy_test.append([x])
newy_train = []
for x in y_train:
    newy_train.append([x])

# 开始多项式特征拟合，以及多元线性回归
from sklearn.linear_model import LinearRegression

prees = []    # 用来存放所用样本在当前对应阶数上的预测值，用于画预测的曲线
for dd in range(1, 7):
    poly_reg = PolynomialFeatures(degree=dd)
    X_ploy_train = poly_reg.fit_transform(x_train)
    X_ploy_test = poly_reg.fit_transform(x_test)
    tt = poly_reg.fit_transform(newx)

    lin2 = LinearRegression()

    lin2.fit(X_ploy_train, newy_train)

    prees.append(lin2.predict(tt))
    predicts = lin2.predict(X_ploy_test)

    rmses.append(np.sqrt(mean_squared_error(y_test, predicts)))
    maes.append(mean_absolute_error(y_test, predicts))
    r2s.append(r2_score(y_test, predicts))

# 画不同阶数的预测曲线
plt.figure(figsize=(15, 10))

for i in range(len(prees)):
    plt.subplot(int('23' + str((i+1))))
    plt.plot(copyx, prees[i])
    plt.ylabel('value')
    plt.xlabel('degree = '+ str(i+1))

plt.savefig('./plots/degree.svg', format='svg')  # 保存为矢量图


# 画不同阶数对应的评价标准的柱状图
plt.figure(figsize=(14, 6))
plt.xlabel('degree')
# r2值图， X轴是参数取值
plt.bar([x-0.15 for x in range(1, 7)], maes, width=0.3, color="green",label='MAEs')
plt.bar([x+0.15 for x in range(1, 7)], rmses, width=0.3, label='RMSEs')
plt.legend(loc='upper left')

plt.savefig('./plots/evaluation_bar.svg', format='svg')  # 保存为矢量图


# 画不同阶数对应的评价标准的折线图
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot([x for x in range(1,7)],maes,marker = "o",markersize=5,markerfacecolor = 'black')
plt.ylabel('mae value')
plt.xlabel('degree')

plt.subplot(132)
plt.plot([x for x in range(1,7)],rmses,marker = "o",markersize=5,markerfacecolor = 'black')
plt.ylabel('rmse value')
plt.xlabel('degree')

plt.subplot(133)
plt.plot([x for x in range(1,7)],r2s,marker = "o",markersize=5,markerfacecolor = 'black')
plt.ylabel('r2 value')
plt.xlabel('degree')

plt.savefig('./plots/evaluation_curve.svg', format='svg') # 保存为矢量图