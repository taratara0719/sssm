import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set(style='darkgrid')

class ParticleFilter(object):
    def __init__(self, y, n_particle):
        self.y = y
        self.n_particle = n_particle
        self.log_likelihood = -np.inf

    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

    def F_inv(self, w_cumsum, idx, u):
            if np.any(w_cumsum < u) == False:
                return 0
            k = np.max(idx[w_cumsum < u])
            return k+1

    def resampling(self, weights):
        w_cumsum = np.cumsum(weights)
        idx = np.asanyarray(range(self.n_particle))
        k_list = np.zeros(self.n_particle, dtype=np.int32) # サンプリングしたkのリスト格納場所

        # 一様分布から重みに応じてリサンプリングする添え字を取得
        for i, u in enumerate(np.random.uniform(0, 1, size=self.n_particle)):
            k = self.F_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list

    def resampling2(self, weights):
        """
        計算量の少ない層化サンプリング
        """
        idx = np.asanyarray(range(self.n_particle))
        u0 = np.random.uniform(0, 1/self.n_particle)
        u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(weights)
        k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
        return k

    
    def simulate(self, seed=71):
        np.random.seed(seed)

        # 時系列データ数
        T = len(self.y)

        #  system model
        #  AR(2)model parameter
        phi1 = 0.529
        phi2 = 0.120
        self.sigma = [np.exp(-2)]*self.n_particle
        a = 1
        F = np.mat([[phi1, phi2, 0], [1, 0, 0], [0, 0, a]])
        self.sigma_ = [self.sigma]

        #  observation model
        H = np.mat([1, 0, 0])
        R = 0.1

        # 潜在変数
        x = np.zeros((3, T+1, self.n_particle))
        x_resampled = np.zeros((3, T+1, self.n_particle))

        # 潜在変数の初期値
        true_x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')
        
        # initial_x =  np.random.normal(0, .01, size=(3,self.n_particle)).T + true_x[0]  #--- (1)
        initial_x = np.zeros((3, self.n_particle)).T + true_x[0]
        x_resampled[:, 0, :] = initial_x.T
        x[:, 0, :] = initial_x.T
        
        
        y_pred = np.zeros((T+1, self.n_particle))

        # 重み
        w        = np.zeros((T, self.n_particle))
        w_normed = np.zeros((T, self.n_particle))

        l = np.zeros(T) # 時刻毎の尤度

        for t in range(T):
            print("\r calculating... t={}".format(t), end="")
            for i in range(self.n_particle):
                # AR(2)モデルを適用
                self.Q = np.mat([[self.sigma[i], 0, 0], [0, 0, 0], [0, 0, 0.01]])
                v = np.random.multivariate_normal([0,0,0], self.Q, 1) # System Noise　#--- (2)
                x[:, t+1, i] = F @ x_resampled[:, t, i] + v # システムノイズの付加
                y_pred[t+1, i] = H @ x[:, t+1, i] 
                w[t, i] = self.norm_likelihood(self.y[t], y_pred[t+1, i], R) # y[t]に対する各粒子の尤度
                
                # TODO: yの時刻合わせる
            w_normed[t] = w[t]/np.sum(w[t]) # 規格化
            l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度
            # Resampling
            #k = self.resampling(w_normed[t]) # リリサンプリングで取得した粒子の添字
            k = self.resampling2(w_normed[t]) # リリサンプリングで取得した粒子の添字（層化サンプリング）
            x_resampled[:, t+1] = x[:, t+1, k]
            self.sigma = np.exp(x_resampled[2, t+1, :])
            self.sigma_.append(self.sigma)

        # 全体の対数尤度
        self.log_likelihood = np.sum(l) - T*np.log(self.n_particle)

        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l

    def get_filtered_value(self, a):
        """
        尤度の重みで加重平均した値でフィルタリングされ値を算出
        """
        return np.diag(np.dot(self.w_normed, self.x[a, 1:].T))

    def draw_graph(self):
        # グラフ描画
        T = len(self.y)
        true_x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')

        plt.subplot(2, 1, 1)
        # plt.figure(figsize=(16,8))
        # plt.plot(range(T), self.y)
        plt.plot(true_x[:, 0], label='true')
        # plt.plot(self.y, label='observed')
        plt.plot(self.get_filtered_value(0), label='x_pred')

        # for t in range(T):
        #     plt.scatter(np.ones(self.n_particle)*t, self.x[0, t], color="r", s=0.1, alpha=0.01)

        plt.title("log likelihood={0:.3f}".format( self.log_likelihood))
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.get_filtered_value(2),label='sigma_pred')
        # plt.plot(np.exp(self.get_filtered_value(2)),label='sigma_pred')
        
        plt.plot(true_x[:, 2], label='true')
        # true_sigma = np.genfromtxt(fname='../data/garch_sigma.txt', delimiter=',')
        # plt.plot(true_sigma, label='true')
        # for t in range(T):
        #     plt.scatter(np.ones(self.n_particle)*t, self.x[2, t], color="r", s=0.1, alpha=0.01)
        plt.legend()

        plt.savefig('../fig/particle_ar2_pred.png')
        print('fig saved')
        plt.show()

x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')
y = np.genfromtxt(fname='../data/garch_obs_states.txt', delimiter=',')

pf = ParticleFilter(y, 1000)
pf.simulate()
pf.draw_graph()
