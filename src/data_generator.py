import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')


def main():
    T = 256  # number of sampling
    r = 1  # number of source components

    np.random.seed(0)
    x = np.zeros((T, r))  # hidden states
    y = np.zeros((T, r))  # observed states
    F = np.zeros((2*r, 2*r))  # parameter
    H = np.ones(r)  # parameter
    Q = np.zeros((2*r, 2*r))  # variance matrix of system noise

    # AR parameter
    phi1 = np.random.normal(0, 0.3, size=r)
    phi2 = np.random.normal(0, 0.2, size=r)

    # initial state

    for i in range(2):
        for j in range(r):
            x[i, j] = np.random.normal(0, 0.5)
            x[i+1, j] = np.random.normal(0, 0.5)

    # observation noise
    obs_noise = np.random.normal(0, 0.2, size=T)

    # system noise
    sigma = 0.5
    sys_noise = np.random.normal(0, sigma, size=(T, r))

    # hidden states
    for i in range(r):
        for t in range(T-2):
            x[t+2,i] = phi1[i] * x[t+1,i] + phi2[i] * x[t,i] + sys_noise[t,i]
    print(x)
    # observed state
    y = x @ H + obs_noise
    #print(y.shape)

    plt.subplot(2, 1, 1)
    for i in range(r):
        plt.plot(x[:,i], label='x{}'.format(i+1))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y, label='y', color='red')
    plt.xlabel('time')
    plt.legend()
    plt.savefig('../fig/states.png')
    print('fig saved')
    
    np.savetxt(fname='../data/hidden_states.txt',fmt='%.5f', X=x, delimiter=',')
    np.savetxt(fname='../data/observed_states.txt',fmt='%.5f', X=y, delimiter=',')
    print('data saved')

    plt.show()

if __name__ == '__main__':
    main()

