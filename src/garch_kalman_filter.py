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
        K_ = [self.K]
        sigma_ = [self.sigma]
        nu_ = [self.nu]
        
        for i in tqdm(range(self.T-1)):
            # prediction
            x_ = self.F @ self.x
            P_ = self.Q + self.F @ self.P @ self.F.T
            
            #filtering
            yi = y[i+1] - self.H @ x_
            S = self.H @ P_ @ self.H.T + self.R
            K = P_ @ self.H.T / S
            self.x = x_ + K * yi
            self.P = P_ - K * self.H @ P_
            self.nu = self.Q - self.Q @ self.H.T /S @ self.H @ self.Q  + K * yi * yi @ K.T

            #  sigma filtering
            if i >= self.p and i >= self.q:
                ar = 0
                ma = 0
                for j in range(1, self.p):
                    ar += self.alpha * math.log(sigma_[-j])
                for k in range(1, self.q):
                    ma += self.beta * math.log(nu_[-k]) 
                s = math.log(sigma0**2) + ar + ma
                self.sigma = math.log(s)
            
            self.X.append(self.x)
            n, m = np.array(np.concatenate(self.X,axis=1))
            if i >= 1:
                self.lse(n)
            self.phi1_.append(self.phi1)
            self.phi2_.append(self.phi2)
        
            yi_.append(yi)
            K_.append(K)
            Q_.append(Q)
            sigma_.append(sigma)
            nu_.append(nu)


        return self
def kf(T, Y, sigma0, x0, P0, F, H, R, p, q, alpha, beta):
    x = x0
    P = P0
    sigma = sigma0
    nu = 0
    yi = 0
    K = 0

    Q = np.mat([[sigma, 0], [0, 0]])
    X = [x]  # hidden states
    yi_ = [yi]  # prediction error
    K_ = [K]
    Q_ = [Q]
    sigma_ = [sigma]
    nu_ = [nu]

    for i in range(T-1):

        # prediction
        x_ = F @ x            # priori state estimate
        P_ = Q + F @ P @ F.T  # priori estimate covariance 
        
        #filtering
        yi = Y[i+1] - H @ x_    # innovation
        S = H @ P_ @ H.T + R    # innovation covariance
        K = P_ @ H.T / S        # Optimal Kalman gain
        x = x_ + K * yi         # posteriori state estimate 
        P = P_ - K * H @ P_     # posteriori estimate covariance
        nu = Q - Q @ H.T /S @ H @ Q  + K * yi * yi @ K.T
        
        if i >= p and i >= q:
            ar = 0
            ma = 0
            for j in range(1, p):
                ar += alpha * math.log(sigma_[-j])
            for k in range(1, q):
                ma += beta * math.log(nu_[-k]) 
            s = math.log(sigma0**2) + ar + ma
            sigma = math.log(s)
            ar = 0
            ma = 0
            

        X.append(x)
        yi_.append(yi)
        K_.append(K)
        Q_.append(Q)
        sigma_.append(sigma)
        nu_.append(nu)

        Q = np.mat([[sigma_[-1], 0], [0, 0]])

    
    print("yi", yi)
    print("x_", x_)
    print("P", P)
    print("P_", P_)
    print("K", K)
    print("H", H)
    

    return X

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

    X = kf(T, y, sigma0, x0, P0, F, H, R, p, q, alpha, beta)
    
    
    plt.plot(x[:, 1], label='x')
    a, b = np.array(np.concatenate(X,axis=1))
    plt.plot(b, label='predicted_x')

    plt.xlabel('time')
    plt.ylabel('x_predict')
    plt.legend()
    

    plt.savefig('../fig/predicted_states.png')
    print('fig saved')
    
    np.savetxt(fname='../data/predicted_x.txt',fmt='%.5f', X=X, delimiter=',')
    #np.savetxt(fname='../data/observed_states2.txt',fmt='%.5f', X=Y, delimiter=',')
    print('data saved')

    plt.show()
    



if __name__ == '__main__':
    main()