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
    sigma = 2
    F = np.mat([[phi1, phi2], [1, 0]])
    Q = np.mat([[sigma, 0], [0, 0]])
    eta = np.zeros((3, 1))

    #  observation model
    H = np.mat([1, 0])
    R = 1

    #  test data generating
    T = 100  # number of sampling
    x = np.mat(np.random.normal(0, 0.3, (2, 1)))
    y = 0
    
    X = [x]
    Y = [y]
    sigma_ = [sigma]
    a = 0.8

    for i in range(T-1):
        nu = np.random.normal(0, 0.3)
        sigma2 = a * np.log(sigma_[-1]) + nu
        sigma = np.exp(sigma2)
        sigma_.append(sigma)

        Q = np.mat([[sigma, 0], [0, 0]])
        # eta[0:2, :] = np.random.multivariate_normal([0,0], Q, 1).T
        # eta[2, :] = np.random.normal(0, 1)
        # x = F @ x + eta
        x = F @ x + np.random.multivariate_normal([0,0], Q, 1).T
        X.append(x)

        y = H @ x + np.random.normal(0, R)
        Y.append(y)
    

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
    np.savetxt(fname='../data/garch_sigma.txt',fmt='%.5f', X=sigma_, delimiter=',')
    print('data saved')

    plt.show()

if __name__ == '__main__':
    main()