import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set(style='darkgrid')

class ParticleFilter(object):
    def __init__(self, y, n_particle, T):
        self.y = y
        self.n_particle = n_particle
        self.log_likelihood = -np.inf
        self.T = T

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


    #  calculator
    def init_initial(self):
        self.phi1_ = [self.phi1]
        self.phi2_ = [self.phi2]
        self.alpha_ = [self.alpha]

        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.g = 0
        self.h = 0
        

    #  phi estimator
    def lse_phi(self, x, x2, log_sigma, log_sigma2):
        self.a += x2[-3]
        self.c += x[-1] * x[-2]
        self.b += x2[-3]
        self.d += x[-1] * x[-2]
        self.e += x[-1] * x[-3]
        # self.f += log_sigma[-1]*log_sigma[-2]
        # self.g += log_sigma2[-2]
        if len(x) <= self.T:
            self.h += (self.y[len(x)-1] - x[-1])**2
        if len(x) >= 3:
            self.phi1 = (self.b * self.c - self.d * self.e) / (self.a * self.b - self.d**2)
            self.phi2 = (self.a * self.e - self.c * self.d) / (self.a * self.b - self.d**2)
            # self.alpha = self.f / self.g
            self.F = np.mat([[self.phi1, self.phi2, 0], [1, 0, 0], [0, 0, self.alpha]])
            self.R = np.sqrt(self.h / len(x))

        return self


    def simulate(self, seed=71):
        np.random.seed(seed)

        # 時系列データ数
        self.T = len(self.y)

        #  system model
        #  AR(2)model parameter
        self.phi1 = 0.128
        self.phi2 = -0.11
        self.sigma = np.exp(0) + np.random.normal(0, 0.01, self.n_particle)
        self.alpha = 1
        self.F = np.mat([[self.phi1, self.phi2, 0], [1, 0, 0], [0, 0, self.alpha]])
        self.sigma_ = [self.sigma]

        self.init_initial()
        
        #  observation model
        H = np.mat([1, 0, 0])
        self.R = 0.1

        # 潜在変数
        x = np.zeros((3, self.T+1, self.n_particle))
        x_resampled = np.zeros((3, self.T+1, self.n_particle))

        # 潜在変数の初期値
        true_x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')
        
        initial_x = np.random.normal(0, 0.01, (3, self.n_particle)).T + true_x[0]
        x_resampled[:, 0, :] = initial_x.T
        x[:, 0, :] = initial_x.T
        
        self.x_mean = 0
        x2_mean = 0
        sigma_mean = 0
        sigma2_mean = 0

        self.x_mean_ = [self.x_mean]
        x2_mean_ = [x2_mean]
        sigma_mean_ = [sigma_mean]
        sigma2_mean_ = [sigma2_mean]
        self.R_ = [self.R]


        y_pre = np.zeros((self.T+1, self.n_particle))

        # 重み
        w        = np.zeros((self.T, self.n_particle))
        w_normed = np.zeros((self.T, self.n_particle))

        l = np.zeros(self.T) # 時刻毎の尤度


        for t in range(self.T):
            print("\r calculating... t={}".format(t), end="")
            for i in range(self.n_particle):
                # AR(2)モデルを適用
                self.Q = np.mat([[self.sigma[i], 0, 0], [0, 0, 0], [0, 0, 0.01]])
                v = np.random.multivariate_normal([0,0,0], self.Q, 1) # System Noise　#--- (2)
                x[:, t+1, i] = self.F @ x_resampled[:, t, i] + v # システムノイズの付加
                y_pre[t+1, i] = H @ x[:, t+1, i] 
                w[t, i] = self.norm_likelihood(self.y[t], y_pre[t+1, i], self.R) # y[t]に対する各粒子の尤度
                
            w_normed[t] = w[t]/np.sum(w[t]) # 規格化
            l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度
            # Resampling
            #k = self.resampling(w_normed[t]) # リリサンプリングで取得した粒子の添字
            k = self.resampling2(w_normed[t]) # リリサンプリングで取得した粒子の添字（層化サンプリング）
            x_resampled[:, t+1] = x[:, t+1, k]
            self.sigma = np.exp(x_resampled[2, t+1, :])
            self.sigma_.append(self.sigma)

            self.x_mean = np.mean(x_resampled[0, t+1, :])
            x2_mean = np.mean(x_resampled[0, t+1, :]**2)
            sigma_mean = np.mean(x_resampled[2, t+1, :])
            sigma2_mean = np.mean(x_resampled[2, t+1, :]**2)

            self.x_mean_.append(self.x_mean)
            x2_mean_.append(x2_mean)
            sigma_mean_.append(sigma_mean)
            sigma2_mean_.append(sigma2_mean)
            

            if t >= 1:
                self.lse_phi(self.x_mean_, x2_mean_, sigma_mean_, sigma2_mean_)
            self.phi1_.append(self.phi1)
            self.phi2_.append(self.phi2)
            self.alpha_.append(self.alpha)
            self.R_.append(self.R)


        # 全体の対数尤度
        self.log_likelihood = np.sum(l) - self.T*np.log(self.n_particle)
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

    def hid_draw_graph(self):
        # グラフ描画
        self.T = len(self.y)
        true_x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')

        plt.subplot(3, 1, 1)
        # plt.figure(figsize=(16,8))
        # plt.plot(range(T), self.y)
        plt.plot(true_x[:, 0], label='true', color='orange')
        # plt.plot(self.y, label='observed')
        plt.plot(self.get_filtered_value(0), label='x_pred')

        # for t in range(T):
        #     plt.scatter(np.ones(self.n_particle)*t, self.x[0, t], color="r", s=0.1, alpha=0.01)

        plt.title("true data estimation")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(true_x[:, 2], label='true',color='orange')
        plt.plot(self.get_filtered_value(2),label='sigma_pred')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.hlines(y = 0.1, xmin = 0, xmax = len(self.y), label = 'true', color='orange')
        plt.plot(self.R_,label='R_pred')
        plt.legend()


        plt.savefig('../fig/garch_r_pred.png')
        print('fig saved')
        plt.show()

    def para_draw_graph(self):
        self.T = len(self.y)

        plt.subplot(2, 1, 1)
        plt.plot(self.phi1_, label='estimate')
        
        plt.title("phi1")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.phi2_, label='estimate')
        
        plt.title("phi2")
        plt.legend()

        # plt.subplot(3, 1, 3)
        # plt.plot(self.alpha_, label='estimate')
        # plt.hlines(y = 1, xmin = 0, xmax = len(self.y), label = 'true', color='orange')
        # plt.title("alpha")
        # plt.legend()

        plt.show()

"""x = np.genfromtxt(fname='../data/garch_hid_states.txt', delimiter=',')"""
y = np.genfromtxt(fname='../data/garch_obs_states.txt', delimiter=',')

pf = ParticleFilter(y, 1000, len(y))
pf.simulate()

print(pf.phi1_[-1])
print(pf.phi2_[-1])

pf.hid_draw_graph()
pf.para_draw_graph()


