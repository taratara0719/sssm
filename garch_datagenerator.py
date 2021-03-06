"""
GARCH性を導入した時系列データの生成
"""
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from tqdm import tqdm
import math

sns.set(style='darkgrid')

def main():
    """
    初期値設定
    システムモデル　x_t = F*x_t-1 + N(0, Q)
    観測モデル　　　y_t = H*x_t + N(0, R)
    """
    np.random.seed(20)
    #  system model
    #  AR(2)model parameter
    phi1 = 0.529
    phi2 = 0.120
    
    sigma = np.exp(0)
    a = 1
    b = 0.01
    F = np.mat([[phi1, phi2, 0], [1, 0, 0], [0, 0, a]])
    Q = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, b]])
    Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])

    #  observation model
    H = np.mat([1, 0, 0])
    R = 0.1

    #  test data generating
    T = 5000  # number of sampling
    x = np.mat(np.random.normal(0, 0.3, (2, 1)))
    z = np.mat([np.log(sigma)])
    x_ = np.vstack([x, z])
    y = 0
    print(x_[-1, 0])
    X = [x_]
    Y = [y]
    sigma_ = [sigma]
    
    """
    データ生成
    """
    for i in range(T-1):
        # ランダムウォークモデル
         x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q, 1).T
         sigma = np.exp(x_[-1, 0])
         sigma_.append(sigma)
         Q = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, b]])
         X.append(x_)
         y = H @ x_ + np.random.normal(0, R)
         Y.append(y)

        # 分散切り替え用モデル
        # if i<= T/2:
        #     x[-1, 0] = np.log(sigma)
        #     Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])
        #     x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q0, 1).T
        #     x[-1, 0] = np.log(sigma)
        #     X.append(x_)
        #     y = H @ x_ + np.random.normal(0, R)
        #     Y.append(y)
        #     sigma_.append(sigma)
        # else:
        #     sigma = np.exp(1)
        #     Q0 = np.mat([[sigma, 0, 0], [0, 0, 0], [0, 0, 0]])
        #     x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q0, 1).T
        #     x_[-1, 0] = np.log(sigma)
        #     X.append(x_)
        #     y = H @ x_ + np.random.normal(0, R)
        #     Y.append(y)
        #     sigma_.append(sigma)

        # 分散一定
        # x_ = F @ x_ + np.random.multivariate_normal([0,0,0], Q0, 1).T
        # X.append(x_)
        # y = H @ x_ + np.random.normal(0, R)
        # Y.append(y)



    plt.subplot(2, 1, 1)
    #for i in range(r):
        #plt.plot(x[:,i], label='x{}'.format(i+1))
    a, b, c = np.array(np.concatenate(X, axis=1))
    plt.plot(a, label='x', linewidth=1)
    plt.legend()
    
    # plt.subplot(3, 1, 2)
    # plt.plot(Y, label='y', color='red', linewidth=0.8)
    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(c, label='sigma', color='orange', linewidth=1)
    plt.xlabel('time')
    plt.legend()


    plt.savefig('../fig/garch_states.png')
    print('fig saved')
    print(X[0])
    np.savetxt(fname='../data/garch_hid_states.txt',fmt='%.5f', X=X, delimiter=',')
    np.savetxt(fname='../data/garch_obs_states.txt',fmt='%.5f', X=Y, delimiter=',')
    np.savetxt(fname='../data/garch_sigma.txt',fmt='%.5f', X=sigma_, delimiter=',')
    print('data saved')

    plt.show()

if __name__ == '__main__':
    main()