import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='darkgrid')

class kf_lse(object):

    def __init__(self, phi1, phi2, T):
        self.phi1 = phi1
        self.phi2 = phi2
        self.T = T

    def init_initial(self, y):
        self.x = np.mat([[0.2936214], [0.67226796]])
        self.P = np.zeros((2, 2))
        self.sigma = 1
        self.Q = np.mat([[self.sigma, 0], [0, 0]])
        self.F = np.mat([[self.phi1, self.phi2], [1, 0]])
        self.H = np.mat([1, 0])
        self.R = 1
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0


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

    def kf(self, y):
        self.init_initial(y)

        self.X = [self.x]
        self.phi1_ = [self.phi1]
        self.phi2_ = [self.phi2]
        
        for i in range(self.T-1):
            # prediction
            x_ = self.F @ self.x
            P_ = self.Q + self.F @ self.P @ self.F.T
            
            #filtering
            yi = y[i+1] - self.H @ x_
            S = self.H @ P_ @ self.H.T + self.R
            K = P_ @ self.H.T / S
            self.x = x_ + K * yi
            self.P = P_ - K * self.H @ P_

            self.X.append(self.x)
            n, m = np.array(np.concatenate(self.X,axis=1))
            if i >= 1:
                self.lse(n)
            self.phi1_.append(self.phi1)
            self.phi2_.append(self.phi2)
            print(i)


        return self
    

def main():
    x = np.genfromtxt(fname='../data/hidden_states2.txt', delimiter=',')
    pred_x = np.genfromtxt(fname='../data/predicted_x.txt', delimiter=',')
    y = np.genfromtxt(fname='../data/observed_states2.txt')
    Y = y.reshape(len(y), 1)

    
    
    #  kalman filtering
    pred = kf_lse(0.6, 0.1, len(x)).kf(Y)

    print(pred.phi1_[-1], pred.phi2_[-1])

    plt.subplot(2, 1, 1)
    a, b = np.array(np.concatenate(pred.X,axis=1))
    plt.plot(x[:, 0], label='x')
    plt.plot(a, label='predicted_x')
    
    plt.title("Hidden states")
    plt.ylabel('x')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(pred.phi1_, label='phi1')
    plt.plot(pred.phi2_, label='phi2')
    plt.title("parameter")
    plt.xlabel('time')
    plt.legend()

    plt.savefig('../fig/kf_lse_pred.png')
    print('fig saved')
    
    plt.show()

if __name__ == '__main__':
    main()