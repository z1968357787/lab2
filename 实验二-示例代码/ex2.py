
import numpy as np
import pandas as pd
import sklearn.datasets as sd
import sklearn.model_selection as sms
import matplotlib.pyplot as plt
import random

# 读取实验训练集和验证集
X_train, y_train = sd.load_svmlight_file('a9a.txt',n_features = 123)
X_valid, y_valid = sd.load_svmlight_file('a9a.t.txt',n_features = 123)


# 将稀疏矩阵转为ndarray类型
X_train = X_train.toarray()#转化为矩阵
X_valid = X_valid.toarray()
y_train = y_train.reshape(len(y_train),1)#转为列向量
y_valid = y_valid.reshape(len(y_valid),1)

#X_train.shape[0]返回数组的大小
X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis = 1)#给x矩阵多增加1列1，拼接在每一行的开头
X_valid = np.concatenate((np.ones((X_valid.shape[0],1)), X_valid), axis = 1)#给x矩阵多增加1列1，拼接在每一行的开头

#定义sigmoid函数，传入参数为向量时，对向量的每个参数进行运算后返回一个列表
def sigmoid(z):
    return 1 / (1 + np.exp(-z))#sigmoid函数

#定义 logistic loss 函数
#X为数据特征
#Y为数据标签
#theta为权重矩阵
def logistic_loss(X, y ,theta):
    hx = sigmoid(X.dot(theta))#求分类值
    cost = np.multiply((1+y), np.log(1+hx)) + np.multiply((1-y), np.log(1-hx))#交叉熵函数
    return -cost.mean()/2

#计算当前 loss
theta = np.zeros((X_train.shape[1],1))#初始化权重系数，获取X_train的列数，即特征值数量
print(logistic_loss(X_train, y_train, theta))



#定义 logistic gradient 函数
#X为数据特征
#Y为数据标签
#theta为权重矩阵
def logistic_gradient(X, y, theta):
    return X.T.dot(sigmoid(X.dot(theta)) - y)

#定义 logistic score 函数，设置阈值为0.5，设置分类阈值
def logistic_score(X, y, theta):
    hx = sigmoid(X.dot(theta))
    hx[hx>=0.5] = 1
    hx[hx<0.5] = -1
    hx = (hx==y)
    return np.mean(hx)

#定义 logistic descent 函数
def logistic_descent(X, y, theta, alpha, num_iters, batch_size, X_valid, y_valid):
    loss_train = np.zeros((num_iters,1))#初始化训练误差矩阵
    loss_valid = np.zeros((num_iters,1))#初始化验证误差矩阵
    data = np.concatenate((y, X), axis=1)#将X与y合并以便分批获取样本
    for i in range(num_iters):
        sample = np.matrix(random.sample(data.tolist(), batch_size))#从数据集中分批(随机获取batch_size大小的样本数量)
        grad = logistic_gradient(sample[:,1:125], sample[:,0], theta)#求解梯度
        theta = theta - alpha * grad#更新参数模型
        loss_train[i] = logistic_loss(X, y, theta)#计算训练集误差
        loss_valid[i] = logistic_loss(X_valid, y_valid, theta)#计算验证集误差
    return theta, loss_train, loss_valid


#执行梯度下降
#全零初始化
theta = np.zeros((X_train.shape[1],1))
alpha = 1e-6
num_iters = 150
opt_theta, loss_train, Lvalidation = logistic_descent(X_train, y_train, theta, alpha, num_iters, 70, X_valid, y_valid)


#选择合适的阈值，将验证集中计算结果大于阈值的标记为正类，反之为负类,得到的平均值
print(logistic_score(X_valid, y_valid, opt_theta))

iteration = np.arange(0, num_iters, step = 1)
fig, ax = plt.subplots(figsize = (12,8))
ax.set_title('Lvalidation change with iteration')
ax.set_xlabel('iteration')#横坐标
ax.set_ylabel('loss')#纵坐标
plt.plot(iteration, Lvalidation, 'r', label='Validation Set Loss')
# plt.plot(iteration, scores, 'g', label='Score on Validation Set')
plt.legend()
plt.show()

#ques2
#定义 hinge loss 函数
#X为数据特征
#Y为数据标签
#theta为权重矩阵
#t为软间隔C
def hinge_loss(X, y, theta, t):
    loss = np.maximum(0, 1 - np.multiply(y, X.dot(theta))).mean()
    reg = np.multiply(theta,theta).sum() / 2
    return t * loss + reg

#计算当前 loss
theta = np.random.random((X_train.shape[1],1))
#设置软间隔为0.5
C = 0.5
print(hinge_loss(X_train, y_train, theta, C))

#定义 hinge gradient 函数 损失函数的梯度
#X为数据特征
#Y为数据标签
#theta为权重矩阵
def hinge_gradient(X, y, theta, C):
    error = np.maximum(0, 1 - np.multiply(y, X.dot(theta))) #此处不加软间隔
    index = np.where(error==0)#为了使得error=0的gw(xi)也等于0，此时需要将gw(xi)=-xiyi中的xi全都设为0，就能保证error=0时，gw(xi)=-xiyi也等于0
    x = X.copy()
    x[index,:] = 0
    grad = theta - C * x.T.dot(y) / len(y)
    #grad[-1] = grad[-1] - theta[-1]
    return grad

#定义 svm decent 函数
def svm_descent(X, y, theta, alpha, num_iters, batch_size, X_valid, y_valid, C): #C是软间隔
    loss_train = np.zeros((num_iters,1))#初始化训练误差矩阵
    loss_valid = np.zeros((num_iters,1))#初始化验证误差矩阵
    data = np.concatenate((y, X), axis=1)#将X与y合并以便分批获取样本
    for i in range(num_iters):
        sample = np.matrix(random.sample(data.tolist(), batch_size))#从数据集中分批(随机获取batch_size大小的样本数量)
        grad = hinge_gradient(sample[:,1:125], sample[:,0], theta, C)#求解梯度
        theta = theta - alpha * grad#参数优化
        loss_train[i] = hinge_loss(X, y, theta, C)#计算训练集误差
        loss_valid[i] = hinge_loss(X_valid, y_valid, theta, C)#计算验证集误差
    return theta, loss_train, loss_valid

#定义 svm score 函数 根据阈值判断是那种类型
def svm_score(X, y, theta):
    hx = X.dot(theta)
    hx[hx>=5] = 1
    hx[hx<5] = -1
    hx = (hx==y)
    return np.mean(hx)

#执行梯度下降
theta = np.random.random((X_train.shape[1],1))
alpha = 1e-6
num_iters = 250
opt_theta, loss_train, Lvalidation = svm_descent(X_train, y_train, theta, alpha, num_iters, 70, X_valid, y_valid, C)


#选择合适的阈值，将验证集中计算结果大于阈值的标记为正类，反之为负类,得到的平均值
print(svm_score(X_valid, y_valid, opt_theta))

iteration = np.arange(0, num_iters, step = 1)
fig, ax = plt.subplots(figsize = (12,8))
ax.set_title('Lvalidation change with iteration')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
plt.plot(iteration, Lvalidation, 'r', label='Validation Set Loss')
plt.legend()
plt.show()




