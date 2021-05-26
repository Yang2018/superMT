class MyMultpleLinearReg:
    def __init__(self, ):
        self.theta = None
        self.J_history = None

    def computerCost(self, X, y, theta):
        m = len(y)
        J = 0

        J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)  # 计算代价J
        return J

    def fit_gd(self, X, y, theta, alpha, num_iters):
        m = len(y)
        n = len(theta)

        temp = np.matrix(np.zeros((n, num_iters)))  # 暂存每次迭代计算的theta，转化为矩阵形式

        J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值

        for i in range(num_iters):  # 遍历迭代次数
            h = np.dot(X, theta)  # 计算内积，matrix可以直接乘
            temp[:, i] = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))  # 梯度的计算
            theta = temp[:, i]
            J_history[i] = self.computerCost(X, y, theta)  # 调用计算代价函数

        self.theta = theta
        self.J_history = J_history

    def GetCoeff(self, ):
        return self.theta, self.J_history

    def predict(self, x_predict):
        # x_predict = np.hstack((np.ones((len(x_predict),1)), x_predict))
        return x_predict.dot(self.theta)
