import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
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

# ↓岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T

    numTestPts = 25
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))

    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T

    return wMat


def predict(xTestMat, wMat):
    return np.dot(xTestMat, wMat.T)

# ↓标准化和归一化函数
def maxmin(data):
    return (data - data.min())/(data.max()-data.min())


def std(data):
    return (data-data.mean())/data.std()

def takey(ll):
    # 用于从分割后的数据中分离出标签与属性
    X = []
    y = []
    for tmp in ll:
        y.append(tmp[-1])
        X.append(tmp[:-1])
    return np.array(X), np.array(y)  # 前一元素是属性，后一元素是标签

data2 = pd.read_excel(r'..\datasets\ENB2012_data.xlsx')
#data2.drop('X6',axis = 1,inplace = True)   # 这步来源于后面通过正则化路径特征选择，去除X6这个特征
X = data2.iloc[:,:-2]
X = maxmin(X)   # 归一化
pd.concat([pd.DataFrame(X),data2['Y1']],axis=1).to_csv('..\datasets\data2.csv')  # 保存预处理好的数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data2['Y1'], test_size=0.2, random_state=1)
X_train = np.hstack((np.ones((len(y_train),1)),X_train))
X_test = np.hstack((np.ones((len(y_test),1)),X_test))
labes = ['nomal','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
ws = ridgeTest(X_train, y_train)
predicts = predict(X_test, ws)

# 画正则化路径图
plt.figure(figsize=(12, 6))
for i in range(0,len(ws.T)):
    plt.plot([x for x in range(0,25)],ws.T[i], label=labes[i])
plt.legend(loc='upper left')
plt.ylabel('ws')
plt.xlabel('numTestPts (λ = e^(numTestPts-10))')

plt.savefig('./plots/regularization.svg', format='svg')  # 保存为矢量图

# 测评岭回归效果
maes = []
r2s= []
mses = []
for x in predicts.T:
    mses.append(mean_squared_error(x,y_test))
    maes.append(mean_absolute_error(x,y_test))
    r2s.append(r2_score(x,y_test))


# MSE值图， X轴是参数取值
plt.figure(figsize=(18, 5))
plt.subplot(131)
plt.plot(np.sqrt(mses))
plt.ylabel('RMSE value')
plt.xlabel('numTestPts (λ = e^(numTestPts-10))')
# r2值图， X轴是参数取值
plt.subplot(132)
plt.plot(r2s)
plt.ylabel('v2 value')
plt.xlabel('numTestPts (λ = e^(numTestPts-10))')
# RMSE值图，
plt.subplot(133)
plt.plot(maes)
plt.ylabel('MAE value')
plt.xlabel('numTestPts (λ = e^(numTestPts-10))')
plt.savefig('./plots/evaluation_curve_R.svg', format='svg')  # save


w = np.exp(7)  # 通过正则化路径和评估值图确定的参数 λ
data = pd.read_csv('..\datasets\data2.csv', index_col = 0)  # 读入之前处理好保存的数据
newdata = cross_validation_split(data,folds=5,Target_Ratio=False,Random_State=1) # 用自己的划分函数进行5则划分

folds = 5
X_train = []
maes = []
r2s = []
mses = []

for x in range(folds):
    # 计算每则的各项得分
    X_test = newdata[x]
    copy = [i for i in range(folds)]
    copy.remove(x)
    for y in copy:
        for z in newdata[y]:
            X_train.append(z)

    x_test, y_test = takey(X_test)
    x_train, y_train = takey(X_train)
    # print(x_train, y_train)

    xMat = np.mat(x_train);
    yMat = np.mat(y_train).T

    cofs = ridgeRegres(xMat, yMat, w)
    # print(cofs)
    predicts = predict(x_test, cofs.T)

    mses.append(mean_squared_error(y_test, predicts))
    maes.append(mean_absolute_error(y_test, predicts))
    r2s.append(r2_score(y_test, predicts))


# 画出每一次验证的得分评价图及平均值
plt.figure(figsize=(15,7))
plt.xlabel('cross_validation_nums (λ = e^(numTestPts-10))')
# r2值图， X轴是参数取值
plt.plot([x for x in range(1,6)], [np.mean(maes)]*5, marker = 'o', markerfacecolor = 'red',\
         markersize=8,color="red",linewidth = 2, linestyle="--",label = 'mean of MAEs')

plt.plot([x for x in range(1,6)], [np.mean(np.sqrt(mses))]*5, marker = 'o', markerfacecolor = 'black',\
         markersize=8,color="black",linewidth =2, linestyle="--", label = 'mean of RMSEs')
plt.bar([x-0.15 for x in range(1,6)], maes, width=0.3 ,color="green",label = 'MAEs')
plt.bar([x+0.15 for x in range(1,6)], np.sqrt(mses), width=0.3 ,label = 'RMSEs')
plt.legend(loc='upper left')
plt.savefig('./plots/cross_va.svg', format='svg')
