import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style='darkgrid')

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
        K = P_ @ H.T @ np.linalg.inv(S)        # Optimal Kalman gain
        x = x_ + K * yi         # posteriori state estimate 
        P = P_ - K * H @ P_     # posteriori estimate covariance
        nu = Q - Q @ H.T @ np.linalg.inv(S) @ H @ Q  + K * yi * yi @ K.T
        
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