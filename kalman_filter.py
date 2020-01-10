import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

def kf(T, Y, x0, P0, F, H, Q, R):
    x = x0
    P = P0

    X = [x]
    

    for i in range(T-1):
        # prediction
        x_ = F @ x
        P_ = Q + F @ P @ F.T
        
        #filtering
        yi = Y[i+1] - H * x_
        S = H * P_ * H.T + R
        K = P_ * H.T * S.I
        x = x_ + K * yi
        P = P_ - K * H * P_
        X.append(x)
    """
    print("yi", yi)
    print("x_", x_)
    print("P", P)
    print("P_", P_)
    print("K", K)
    print("H", H)
    """

    return X

def main():

    x = np.genfromtxt(fname='../data/hidden_states2.txt', delimiter=',')
    y = np.genfromtxt(fname='../data/observed_states2.txt')
    Y = y.reshape(len(y), 1)

    r = 1  # number of source components

    np.random.seed(0)
    #  system model
    #  AR(2)model parameter
    phi1 = np.random.normal(0, 0.3)
    phi2 = np.random.normal(0, 0.3)
    sigma = 1
    F = np.mat([[phi1, phi2], [1, 0]])  
    Q = np.mat([[sigma, 0], [0, 0]])
    print(phi1)
    print(phi2)

    #  observation model
    c = 1
    H = np.mat([c, 0])
    R = 1

    x0 = np.random.normal(0, 0.3, (2, 1))
    P0 = np.zeros((2, 2))
    print(x0)
    #  kalman filtering
    X = kf(len(x), Y, x0, P0, F, H, Q, R)
    
    #  グラフの描画
    
    plt.plot(x[:, 0], label='x')
    plt.plot(Y, label='y')

    a, b = np.array(np.concatenate(X,axis=1))
    plt.plot(a, label='predicted_x')
    

    plt.title("Hidden states")
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend()
    

    plt.savefig('../fig/predicted_states.png')
    print('fig saved')
    
    np.savetxt(fname='../data/predicted_x.txt',fmt='%.5f', X=X, delimiter=',')
    #np.savetxt(fname='../data/observed_states2.txt',fmt='%.5f', X=Y, delimiter=',')
    print('data saved')

    plt.show()
    



if __name__ == '__main__':
    main()