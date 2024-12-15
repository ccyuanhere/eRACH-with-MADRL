class Config:
    def __init__(self):
        self.J = 5 #UT数量
        self.K = 2 #轨道平面数量
        self.P = 2 #前导码数量
        self.N = 200 #RA机会数
        self.rho = 1.0 #平衡吞吐量和冲突率的系数
        self.mu = 0.0 #g()归一化的均值
        self.sigma = 1.0 #g()归一化的缩放
        self.action_dim = self.K + 1  #动作维度：BACKOFF + K个轨道平面
        self.state_dim = 6 #状态维度：[n, a_{j}[n-1], R_j, c_j, sat_features(K维)]
        self.learning_rate = 1e-3 #学习率
        self.gamma = 0.99 #折扣因子
        self.beta_e = 0.01 #熵正则化系数
