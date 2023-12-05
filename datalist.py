import numpy as np
import numpy.linalg as LA
import math

class Test():
    def __init__(self, size):
        self.mean0 = np.zeros(5)
        self.mean1 = np.array([0, 5, 5, 5, 0])
        self.cov0 = np.eye(5)
        self.cov1 = np.array([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]])
        self.size = size
        self.columns = ["Out 0", "Out 1", "Out 2", "Out 3", "Out 4"]
        self.dim = 5
    
    def true_KL(self):
        true_KL = (math.log(LA.det(self.cov0)/LA.det(self.cov1)) + np.trace(LA.inv(self.cov0)@self.cov1) 
                    + ((self.mean1.T - self.mean0.T) @ LA.inv(self.cov0) @ (self.mean1 - self.mean0)) - 5) / 2
        print(f"true KL:{true_KL}")
    
    def gen_data(self):
        data_0 = np.random.multivariate_normal(self.mean0, self.cov0, size=self.size//2, check_valid="raise")
        data_1 = np.random.multivariate_normal(self.mean1, self.cov1, size=self.size//2, check_valid="raise")
        data = np.concatenate([data_0, data_1])
        target = np.concatenate([np.zeros(self.size//2), np.ones(self.size//2)])
        return data, target

class Normal():
    def __init__(self, size):
        self.mean_x = np.array([0, 0, 2, 0, 0])
        self.cov_x = np.array([[1, 0, 0, 0, 0],
                               [0, 3, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 4, 0],
                               [0, 0, 0, 0, 1]])
        self.mean_eps = np.array([0, 0, 0, 0, 0])
        self.cov_eps = 1
        self.trans0 = np.array([[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
        self.trans1 = np.array([[1, 0, 0, 0, 0],
                                [0, 3, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
        self.size = size
        self.columns = ["Feature 0", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Out 0", "Out 1", "Out 2", "Out 3", "Out 4"]
        self.in_dim = 5
        self.out_dim = 5
    
    def true_KL(self):
        trans_diff = self.trans1 - self.trans0
        true_KL = (np.trace((trans_diff @ self.cov_x @ trans_diff.T)) + np.linalg.norm((trans_diff @ self.mean_x.T), ord=2)**2)/(2*self.cov_eps)
        print(f"true KL:{true_KL.sum()}")
        
    def gen_data(self):
        x = np.random.multivariate_normal(self.mean_x, self.cov_x, size=self.size, check_valid="raise")
        r = np.random.binomial(self.size,0.5)
        eps = np.random.multivariate_normal(self.mean_eps, self.cov_eps*np.eye(5), size=self.size, check_valid="raise")
        y0 = (self.trans0 @ x[:r].T + eps[:r].T).T
        data_0 = np.concatenate([x[:r], y0], axis=1)
        y1 = (self.trans1 @ x[r:].T + eps[r:].T).T
        data_1 = np.concatenate([x[r:], y1], axis=1)
        data = np.concatenate([data_0, data_1])
        target = np.concatenate([np.zeros(r), np.ones(self.size-r)])
        return data, target

class Const():
    def __init__(self, size):
        self.mean_x = np.array([0, 0, 2, 0, 0])
        self.cov_x = np.array([[1, 0, 0, 0, 0],
                               [0, 3, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 4, 0],
                               [0, 0, 0, 0, 1]])
        self.mean_eps = np.array([0, 0, 0, 0, 0])
        self.cov_eps = 1
        self.trans = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]])
        self.const0 = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        self.const1 = np.array([5, 2, 3, 4, 5]).reshape(-1, 1)
        self.size = size
        self.columns = ["Feature 0", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Out 0", "Out 1", "Out 2", "Out 3", "Out 4"]
        self.in_dim = 5
        self.out_dim = 5
    
    def true_KL(self):
        const_diff = self.const1 - self.const0
        true_KL = np.linalg.norm(const_diff, ord=2)**2 / (2 * self.cov_eps)
        print(f"true KL:{true_KL.sum()}")
    
    def gen_data(self):
        x = np.random.multivariate_normal(self.mean_x, self.cov_x, size=self.size, check_valid="raise")
        r = np.random.binomial(self.size,0.5)
        eps = np.random.multivariate_normal(self.mean_eps, self.cov_eps*np.eye(5), size=self.size, check_valid="raise")
        y0 = (self.trans @ x[:r].T + self.const0 + eps[:r].T).T
        data_0 = np.concatenate([x[:r], y0], axis=1)
        y1 = (self.trans @ x[r:].T + self.const1 + eps[r:].T).T
        data_1 = np.concatenate([x[r:], y1], axis=1)
        data = np.concatenate([data_0, data_1])
        target = np.concatenate([np.zeros(r), np.ones(self.size-r)])
        return data, target

class NonRCT():
    def __init__(self, size):
        self.mean_x = np.array([0, 0, 2, 0, 0])
        self.cov_x = np.array([[1, 0, 0, 0, 0],
                               [0, 3, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 4, 0],
                               [0, 0, 0, 0, 1]])
        self.mean_eps = np.array([0, 0, 0, 0, 0])
        self.cov_eps = 1
        self.alpha = np.array([1, 2, 3, 4, 5])
        self.beta = 2
        self.trans0 = np.array([[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
        self.trans1 = np.array([[1, 0, 0, 0, 0],
                                [0, 3, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
        self.size = size
        self.columns = ["Feature 0", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Out 0", "Out 1", "Out 2", "Out 3", "Out 4"]
        self.in_dim = 5
        self.out_dim = 5
    
    def true_KL(self):
        trans_diff = self.trans1 - self.trans0
        true_KL = (np.trace((trans_diff @ self.cov_x @ trans_diff.T)) + np.linalg.norm((trans_diff @ self.mean_x.T), ord=2)**2)/(2*self.cov_eps)
        print(f"true KL:{true_KL.sum()}")
    
    def bernoulli(self, x):
        return np.random.binomial(1, x)
    
    def assign(self, x):
        x = self.alpha @ x + self.beta
        p = 1/(1+np.exp(-x))
        ber = np.vectorize(self.bernoulli)
        return ber(p)

    def gen_data(self):
        x = np.random.multivariate_normal(self.mean_x, self.cov_x, size=self.size, check_valid="raise")
        t = self.assign(x.T)
        treat = x[t==1]
        control = x[t==0]
        control_size = control.shape[0]
        eps = np.random.multivariate_normal(self.mean_eps, self.cov_eps*np.eye(5), size=self.size, check_valid="raise")
        y0 = (self.trans0 @ control.T + eps[:control_size].T).T
        data_0 = np.concatenate([control, y0], axis=1)
        y1 = (self.trans1 @ treat.T + eps[control_size:].T).T
        data_1 = np.concatenate([treat, y1], axis=1)
        data = np.concatenate([data_0, data_1])
        target = np.concatenate([np.zeros(control_size), np.ones(self.size-control_size)])
        return x, t, data, target

def make(s, size):
    if s == "test":
        return Test(size)
    elif s == "normal":
        return Normal(size)
    elif s == "const":
        return Const(size)
    elif s == "nonRCT":
        return NonRCT(size)