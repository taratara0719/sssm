import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm

sns.set(style='darkgrid')

class kf_lse(object):

    def __init__(self, phi1, phi2, alpha, T):
        self.phi1 = phi1
        self.phi2 = phi2
        self.alpha = alpha
        self.T = T
    
    #  初期値
    def init_initial(self, y):
        #  hiddenn observed parameter
        # self.x = np.mat([[0.2936214], [0.67226796]])
        self.x = np.mat([[0.3], [0.9], [1]])
        self.P = np.random.normal(0, 1, (3, 3))
        self.sigma = 1
        self.Q = np.mat([[self.sigma, 0, 0], [0, 0, 0], [0, 0, 0.1]])
        self.F = np.mat([[self.phi1, self.phi2, 0], [1, 0, 0], [0, 0, self.alpha]])
        self.H = np.mat([1, 0, 1])
        self.R = 1

        #  calculator
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.g = 0
        self.h = 0
        

    #  phi estimator
    def lse_phi(self, x, sigma):
        self.a += x[-3]**2
        self.c += x[-1] * x[-2]
        self.b += x[-3]**2
        self.d += x[-1] * x[-2]
        self.e += x[-1] * x[-3]
        self.f += sigma[-1]*sigma[-2]
        self.g += sigma[-2]**2
        if len(x) >= 3:
            self.phi1 = (self.b * self.c - self.d * self.e) / (self.a * self.b - self.d**2)
            self.phi2 = (self.a * self.e - self.c * self.d) / (self.a * self.b - self.d**2)
            self.alpha = self.f / self.g
            self.F = np.mat([[self.phi1, self.phi2, 0], [1, 0, 0], [0, 0, self.alpha]])

        return self
        

    #  kalman filter
    def kf(self, y):
        self.init_initial(y)

        self.X = [self.x]
        self.phi1_ = [self.phi1]
        self.phi2_ = [self.phi2]
        self.alpha_ = [self.alpha]
        self.sigma_ = [self.sigma]

        self.sigma = np.exp(self.x[-1, 0])
        self.sigma_.append(self.sigma)
        self.Q = np.mat([[self.sigma, 0, 0], [0, 0, 0], [0, 0, 0.1]])

        for self.i in tqdm(range(self.T-1)):
            # prediction
            x_ = self.F @ self.x
            P_ = self.Q + self.F @ self.P @ self.F.T
            
            #filtering
            yi = y[self.i+1] - self.H @ x_
            S = self.H @ P_ @ self.H.T + self.R
            self.K = P_ @ self.H.T / S
            self.x = x_ + self.K * yi
            self.P = P_ - self.K * self.H @ P_
                
            #  phi iteration
            self.sigma = np.exp(self.x[-1, 0])
            self.X.append(self.x)
            self.sigma_.append(self.sigma)
            self.Q = np.mat([[self.sigma, 0, 0], [0, 0, 0], [0, 0, 0.1]])
            
            n, m, l = np.array(np.concatenate(self.X,axis=1))

            if self.i >= 1:
                self.lse_phi(n, l)
                
            self.phi1_.append(self.phi1)
            self.phi2_.append(self.phi2)
            self.alpha_.append(self.alpha)

        return self
    

def main():

    x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')
    y = np.genfromtxt(fname='../data/garch_obs_states.txt', delimiter=',')
    sigma = np.genfromtxt(fname='../data/garch_sigma.txt', delimiter=',')
    
    np.random.seed(0)
    pred = kf_lse(0.5, 0.1, 0.7, len(x)).kf(y)
    print(pred.alpha_[-1])
    print(pred.phi1_[-1])
    print(pred.phi2_[-1])
    print(len(pred.sigma_))
    print(len(pred.X))

    
    plt.subplot(4, 1, 1)
    a, b, c = np.array(np.concatenate(pred.X,axis=1))
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

    plt.subplot(4, 1, 4)
    plt.plot(pred.sigma_, label='pred')
    plt.plot(sigma, label='true')
    plt.title("sigma")
    plt.xlabel('time')
    plt.legend()

    plt.savefig('../fig/garch_pred.png')
    print('fig saved')
    
    plt.show()
    



if __name__ == '__main__':
    main()