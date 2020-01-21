import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

sns.set(style='darkgrid')

def main():
    r = 1  # number of source components

    np.random.seed(0)
    #  system model
    #  AR(2)model parameter
    phi1 = np.random.normal(0, 0.3)
    phi2 = np.random.normal(0, 0.3)
    sigma = 1
    F = np.mat([[phi1, phi2], [1, 0]])
    Q = np.mat([[sigma, 0], [0, 0]])

    #  observation model
    c = 1
    H = np.mat([c, 0])
    R = 1

    #  test data generating
    T = 200  # number of sampling
    x = np.mat(np.random.normal(0, 0.3, (2, 1)))
    y = 0
    
    X = [x]
    Y = [y]
    sigma_ = [sigma]
    nu = 1
    nu_ = [nu]
    alpha = 0.7
    beta = 0.4
    p = 1
    q = 0

    for i in range(T-1):
        Q = np.mat([[sigma, 0], [0, 0]])
        x = F @ x + np.random.multivariate_normal([0, 0], Q, 1).T
        #x_ = np.hstack((X, x))
        X.append(x)
        y = H @ x + np.random.normal(0, R)
        Y.append(y)
        nu = np.random.normal(0, 2)
        sigma = sigma + np.random.normal(0, 1)
        nu_.append(nu)
        sigma_.append(sigma)
        # sigma_.append(sigma)
        # if i >= p and i >= q:
        #         ar = 0
        #         ma = 0
        #         for j in range(1, p):
        #             ar += alpha * np.log(sigma_[-j])
        #         for k in range(1, q):
        #             ma += beta * np.log((nu_[-k])**2) 
        #         s = np.log(sigma_[0]) + ar + ma
        #         sigma = np.exp(s)
        # sigma = (sigma_[0]*(sigma_[-1])**alpha*(nu_[-1])**beta)
        
        
    

    plt.subplot(3, 1, 1)
    #for i in range(r):
        #plt.plot(x[:,i], label='x{}'.format(i+1))
    a, b = np.array(np.concatenate(X, axis=1))
    plt.plot(a, label='x')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(Y, label='y', color='red')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(sigma_, label='sigma', color='orange')
    plt.xlabel('time')
    plt.legend()


    plt.savefig('../fig/garch_states.png')
    print('fig saved')
    
    np.savetxt(fname='../data/garch_hid_states.txt',fmt='%.5f', X=X, delimiter=',')
    np.savetxt(fname='../data/garch_obs_states.txt',fmt='%.5f', X=Y, delimiter=',')
    print('data saved')

    plt.show()

if __name__ == '__main__':
    main()