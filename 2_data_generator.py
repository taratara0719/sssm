import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    R = 0.3

    #  test data generating
    T = 10000  # number of sampling
    x = np.mat(np.random.normal(0, 0.3, (2, 1)))
    y = 0
    
    X = [x]
    Y = [y]
    for i in range(T-1):
        x = F @ x + np.random.multivariate_normal([0, 0], Q, 1).T
        #x_ = np.hstack((X, x))
        X.append(x)
        y = H @ x + np.random.normal(0, R)
        Y.append(y)
    

    plt.subplot(2, 1, 1)
    #for i in range(r):
        #plt.plot(x[:,i], label='x{}'.format(i+1))
    a, b = np.array(np.concatenate(X, axis=1))
    plt.plot(a, label='x')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(Y, label='y', color='red')
    plt.xlabel('time')
    plt.legend()

    plt.savefig('../fig/states2.png')
    print('fig saved')
    
    np.savetxt(fname='../data/hidden_states2.txt',fmt='%.5f', X=X, delimiter=',')
    np.savetxt(fname='../data/observed_states2.txt',fmt='%.5f', X=Y, delimiter=',')
    print('data saved')

    plt.show()

if __name__ == '__main__':
    main()