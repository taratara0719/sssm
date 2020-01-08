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


def main():
    
    x = np.genfromtxt(fname='../data/hidden_states2.txt', delimiter=',')
    pred_x = np.genfromtxt(fname='../data/predicted_x.txt', delimiter=',')

    np.random.seed(0)
    phi1 = np.random.normal(0, 0.3)
    phi2 = np.random.normal(0, 0.3)

    pred_phi_true_x = lse(x[:, 0])
    pred_phi_pred_x = lse(pred_x[:, 0])

    print("pred_phi_true_x", pred_phi_true_x)
    print("pred_phi_pred_x", pred_phi_pred_x)
    print("true_phi         ", phi1, phi2)
    print(x.shape)
    
    


    

if __name__ == '__main__':
    main()