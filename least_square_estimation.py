import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

def lse(x):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    
    for i in range(2, len(x)):
        
        a += x[i-1]**2
        b += x[i-2]**2
        c += x[i] * x[i-1]
        d += x[i-1] * x[i-2]
        e += x[i] * x[i-2]
    
    phi1 = (b * c - d * e) / (a * b - d**2)
    phi2 = (a * e - c * d) / (a * b - d**2)

    return phi1, phi2

def kf(T, Y, x0, P0, F, H, Q, R):
    x = x0
    P = P0

    X = [x]
    

    for i in range(T-1):
        # prediction
        x_ = F @ x
        P_ = Q + F @ P @ F.T
        
        #filtering
        yi = Y[i+1] - H @ x_
        S = H @ P_ @ H.T + R
        K = P_ @ H.T / S
        x = x_ + K * yi
        P = P_ - K * H @ P_
        X.append(x)
        phi = lse(x)

    return X

def main():
    x = np.genfromtxt(fname='../data/hidden_states2.txt', delimiter=',')
    pred_x = np.genfromtxt(fname='../data/predicted_x.txt', delimiter=',')
    y = np.genfromtxt(fname='../data/observed_states2.txt')
    Y = y.reshape(len(y), 1)

    phi1 = 0.6
    phi2 = 0.1
    sigma = 1
    F = np.mat([[phi1, phi2], [1, 0]])  
    Q = np.mat([[sigma, 0], [0, 0]])
    

    #  observation model
    c = 1
    H = np.mat([c, 0])
    R = 0.3

    x0 = np.mat([[0.2936214], [0.67226796]])
    P0 = np.zeros((2, 2))

    #  kalman filtering
    X = kf(len(x), Y, x0, P0, F, H, Q, R)

if __name__ == '__main__':
    main()