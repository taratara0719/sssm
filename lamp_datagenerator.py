import numpy as np
"""import matplotlib as mpl
mpl.use('Agg')"""
import matplotlib.pyplot as plt
import seaborn as sns
#from tqdm import tqdm
import math

sns.set(style='darkgrid')

def main():
    np.random.seed(20)
    #  system model
    #  AR(2)model parameter
    phi1 = 0.529
    phi2 = 0.120
    
    sigma = np.exp(0)
    a = 1
    b = 0.001
    F = np.mat([[phi1, phi2, 0], [1, 0, 0], [0, 0, a]])
    Q = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, b]])

    Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])



    #  observation model
    H = np.mat([1, 0, 0])
    R = 0.1

    #  test data generating
    T = 5000  # number of sampling
    x = np.random.normal(0, 0.3, (1, 2))
    z = np.log(sigma)
    x_ = np.append(x, z).reshape(3, 1)
    y = 0
    print(x_)
    X = [x_]
    Y = [y]
    sigma_ = [sigma]
    

    for i in range(T-1):
        if i<= T//3:
            x[-1, 0] = 0
            Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])
            x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q0, 1).T
            X = np.append(X, x_)
            y = H @ x_ + np.random.normal(0, R)
            Y = np.append(Y, y)
            sigma_ = np.append(sigma_, sigma)
        elif i <= (T*2)//3:
            sigma = np.exp(9/T*i - 3)
            x_[-1, 0] = 9/T*i - 3
            Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])
            x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q0, 1).T
            X = np.append(X, x_)
            y = H @ x_ + np.random.normal(0, R)
            Y = np.append(Y, y)
            sigma_ = np.append(sigma_, sigma)
        else:
            sigma = np.exp(3)
            Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])
            x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q0, 1).T
            x_[-1, 0] = np.log(sigma)
            X = np.append(X, x_)
            y = H @ x_ + np.random.normal(0, R)
            Y = np.append(Y, y)
            sigma_ = np.append(sigma_, sigma)


    X = X.reshape(T, 3)

    plt.subplot(3, 1, 1)
    #for i in range(r):
        #plt.plot(x[:,i], label='x{}'.format(i+1))
    #a, b = np.array(np.concatenate(X, axis=1))
    plt.plot(X[:, 0], label='x', linewidth=1)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(Y, label='y', color='red', linewidth=1)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(np.log(sigma_), label='sigma', color='orange', linewidth=1)
    plt.xlabel('time')
    plt.legend()


    plt.savefig('../fig/garch_states.png')
    print('fig saved')
    

    np.savetxt(fname='../data/lamp_hid_states.txt',fmt='%.5f', X=X, delimiter=',')
    np.savetxt(fname='../data/lamp_obs_states.txt',fmt='%.5f', X=Y, delimiter=',')
    np.savetxt(fname='../data/lamp_sigma.txt',fmt='%.5f', X=np.log(sigma_), delimiter=',')
    print('data saved')

    plt.show()

if __name__ == '__main__':
    main()