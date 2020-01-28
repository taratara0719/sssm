import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm

sns.set(style='darkgrid')

class kf_lse(object):

    def __init__(self, phi1, phi2, p, q, T):
        self.phi1 = phi1
        self.phi2 = phi2
        self.p = p
        self.q = q
        self.T = T
    
    #  初期値
    def init_initial(self, y):
        #  hiddenn observed parameter
        # self.x = np.mat([[0.2936214], [0.67226796]])
        self.x = np.mat([[0.3], [0.9]])
        self.P = np.zeros((2, 2))
        self.sigma = 1
        self.Q = np.mat([[self.sigma, 0], [0, 0]])
        self.F = np.mat([[self.phi1, self.phi2], [1, 0]])
        self.H = np.mat([1, 0])
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


        self.nu = np.zeros((2, 1))
        self.K = 0
        self.v = np.zeros((2, 1))
        self.alpha = 0.4
        self.beta = 0.5
    

    #  phi estimator
    def lse_phi(self, x):
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

    #  q = 1 beta estimator
    def lse_beta(self, sigma_, v):
        self.f += np.log(sigma_[0])*np.log(v[-2])
        self.g += np.log(sigma_[-1])*np.log(v[-2])
        self.h += np.log(v[-2])
        if len(v) >= 1:
            self.beta = (self.f + self.g) / self.h**2
                
        return self    

    def lse_alpha(self, sigma_):
        self.f += sigma_[0]*sigma_[-2]
        self.g += sigma_[-1]*sigma_[-2]
        self.h += sigma_[-2]
        self.alpha = (self.f + self.g) / self.h**2
                
        return self    
        
            

    #  kalman filter
    def kf(self, y):
        self.init_initial(y)

        self.X = [self.x]
        self.phi1_ = [self.phi1]
        self.phi2_ = [self.phi2]
        self.alpha_ = [self.alpha]
        self.beta_ = [self.beta]
        self.K_ = [self.K]
        self.sigma_ = [self.sigma]
        self.nu_ = [self.nu]
        self.Q_ = [self.Q]
        self.v_ = [self.v]

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
            
            #　残差
            #  厳密解
            self.nu = self.Q - self.Q * self.H.T / S * self.H * self.Q  + self.K * yi * yi * self.K.T
            #  approximation
            # self.nu = np.power(self.K * yi, 2)
            #  真値を用いる
            # self.nu = np.abs(x_true[self.i] - self.x[0])

            self.K_.append(self.K)
            self.Q_.append(self.Q)
            self.nu_.append(self.nu[0, 0])

            #  sigma filtering
            if self.i >= self.p and self.i >= self.q:
                self.ar = 0
                self.ma = 0
                for j in range(1, self.p+1):
                    self.ar += self.alpha * np.log(self.sigma_[-j])
                for k in range(1, self.q+1):
                    self.ma += self.beta * np.log(self.nu_[-k])
                log_sigma =  np.log(self.sigma) + self.ar + self.ma
                self.sigma = np.exp(log_sigma)
                
            #  phi iteration
            self.X.append(self.x)
            self.sigma_.append(self.sigma)
            n, m = np.array(np.concatenate(self.X,axis=1))
            
            if self.i >= 1:
                self.lse_phi(n)
                # self.lse_alpha(self.sigma_)
                self.lse_beta(self.sigma_, self.nu_)
            self.phi1_.append(self.phi1)
            self.phi2_.append(self.phi2)
            self.alpha_.append(self.alpha)
            self.beta_.append(self.beta)
            self.Q = np.mat([[self.sigma, 0], [0, 0]])
        return self
    

def main():

    x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')
    y = np.genfromtxt(fname='../data/garch_obs_states.txt', delimiter=',')
    sigma = np.genfromtxt(fname='../data/garch_sigma.txt', delimiter=',')
    
    pred = kf_lse(0.5, 0.1, 0, 1, len(x)).kf(y)
    print(pred.sigma_[-1])
    print(pred.phi1_[-1])
    print(pred.phi2_[-1])

    
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