import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm

sns.set(style='darkgrid')

class kf_lse(object):

    def __init__(self, phi1, phi2, p, q, alpha, beta, T):
        self.phi1 = phi1
        self.phi2 = phi2
        self.p = p
        self.q = q
        self.T = T
        self.alpha = alpha
        self.beta = beta
    
    #  初期値
    def init_initial(self, y):
        self.x = np.mat([[0.2936214], [0.67226796]])
        self.P = np.zeros((2, 2))
        self.sigma = 1
        self.sigma0 = 1
        self.Q = np.mat([[self.sigma, 0], [0, 0]])
        self.F = np.mat([[self.phi1, self.phi2], [1, 0]])
        self.H = np.mat([1, 0])
        self.R = 1
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.nu = 0
        self.K = 0
    

    #  最小二乗法
    def lse(self, x):
        
        self.a += x[-3]**2
        self.c += x[-1] * x[-2]
        self.b += x[-3]**2
        self.d += x[-1] * x[-2]
        self.e += x[-1] * x[-3]
        if len(x) >= 3:
            self.phi1 = (self.b * self.c - self.d * self.e) / (self.a * self.b - self.d**2)
            self.phi2 = (self.a * self.e - self.c * self.d) / (self.a * self.b - self.d**2)
            self.F = np.mat([[self.phi1, self.phi2], [1, 0]])

        return self

    #  カルマンフィルター
    def kf(self, y):
        self.init_initial(y)

        self.X = [self.x]
        self.phi1_ = [self.phi1]
        self.phi2_ = [self.phi2]
        self.K_ = [self.K]
        self.sigma_ = [self.sigma]
        self.nu_ = [self.nu]
        self.Q_ = [self.Q]
        
        for i in tqdm(range(self.T-1)):
            # prediction
            x_ = self.F @ self.x
            P_ = self.Q + self.F @ self.P @ self.F.T
            
            #filtering
            yi = y[i+1] - self.H @ x_
            S = self.H @ P_ @ self.H.T + self.R
            self.K = P_ @ self.H.T / S
            self.x = x_ + self.K * yi
            self.P = P_ - self.K * self.H @ P_
            self.nu = self.Q - self.Q * self.H.T /S * self.H * self.Q  + self.K * yi * yi * self.K.T

            self.K_.append(self.K)
            self.Q_.append(self.Q)
            self.nu_.append(self.nu)

            #  sigma filtering
            if i >= self.p and i >= self.q:
                ar = 0
                ma = 0
                for j in range(1, self.p):
                    ar += self.alpha * np.log(self.sigma_[-j])
                for k in range(1, self.q):
                    ma += self.beta * np.log(self.nu_[-k]) 
                s = np.log(self.sigma_[0]) + ar + ma
                self.sigma = np.exp(s)
            self.sigma_.append(self.sigma)
            
            #  phi iteration
            self.X.append(self.x)
            n, m = np.array(np.concatenate(self.X,axis=1))
            if i >= 1:
                self.lse(n)
            self.phi1_.append(self.phi1)
            self.phi2_.append(self.phi2)
            
        return self
    

def main():

    x = np.genfromtxt(fname='../data/hidden_states2.txt', delimiter=',')
    y = np.genfromtxt(fname='../data/observed_states2.txt')
    Y = y.reshape(len(y), 1)
    
    x0 = np.zeros((2, 1))
    P0 = np.zeros((2, 2))

    T = 256
    r = 1  # number of source components

    np.random.seed(0)
    #  system model
    #  AR(2)model parameter
    phi1 = np.random.normal(0, 0.3)
    phi2 = np.random.normal(0, 0.3)
    F = np.mat([[phi1, phi2], [1, 0]])  
    sigma0 = 0.5
    
    #  observation model
    c = 1
    H = np.mat([c, 0])
    R = np.mat([[0.4, 0], [0, 0]])

    p = 0
    q = 2
    alpha = 0
    beta = 0.5

    pred = kf_lse(0.529, 0.120, 0, 2, 0, 0.5, len(x)).kf(y)
    
    
    plt.subplot(4, 1, 1)
    a, b = np.array(np.concatenate(pred.X,axis=1))
    plt.plot(x[:, 0], label='x')
    plt.plot(a, label='predicted_x')
    
    plt.title("Hidden states")
    plt.ylabel('x')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(pred.phi1_, label='estimate')
    plt.hlines(y = 0.529, xmin = 0, xmax = len(x[:, 0]), label = 'true', color='orange')
    plt.title("phi1")
    plt.xlabel('time')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(pred.phi2_, label='estimate')
    plt.hlines(y = 0.120, xmin = 0, xmax = len(x[:, 0]), label = 'true', color='orange')
    plt.title("phi2")
    plt.xlabel('time')
    plt.legend()

    # plt.subplot(4, 1, 4)
    # plt.plot(pred.sigma_, label='sigma')
    # plt.xlabel('time')
    # plt.legend()

    plt.savefig('../fig/garch_pred.png')
    print('fig saved')
    
    plt.show()
    



if __name__ == '__main__':
    main()