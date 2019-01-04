import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set(color_codes = True)
from itertools import product
#%%

class GPR(object):
    
    #TODO: add kernels
    
    @classmethod
    def mix1(cls, x, y, length=1.0, period= 1, const1 = 1.0, const2 = 1.0):
        return cls.kernel_gaussian(x,y,length) + cls.kernel_periodic(x,y,length,period)
    
    @classmethod #maybe update params
    def kernel_gaussian(cls, x, y, length=1.0, const = 1.0):
        return const*np.exp(-0.5 * np.power(x-y, 2) / length)
    
    @classmethod
    def kernel_laplacian(cls, x,y,length=1, const = 1.0):
        return np.exp(-0.5*np.abs(x-y) / length)
    
    @classmethod
    def kernel_periodic(cls, x,y,length=1, period=1, const = 1.0):
        sin_argument = np.pi*np.abs(x-y)/period
        exp_argument = -2*np.power(np.sin(sin_argument),2)/length
        return const*np.exp(exp_argument)
    
    @classmethod #maybe update params
    def generate_kernel(cls, kernel, length=1, period=1, const1 = 1.0, const2 = 1.0):
        def wrapper(*args, **kwargs):
            kwargs.update({"length": length})
            kwargs.update({"const": const1})
            if kernel == cls.kernel_periodic or kernel == cls.mix1:
                kwargs.update({"period": period})
            return kernel(*args, **kwargs)
        return wrapper
    
    def __init__(self, x, y, kernel=None, R=0):
        super().__init__()
        self.x = x
        self.y = y
        self.N = len(self.x)
        self.R = R
        
        self.K = []
        self.mean = []
        self.kernel = kernel if kernel else self.kernel_gaussian
        self.setup_K()
        
    @classmethod
    def calculate_K(cls, x, kernel, R=0):
        N = len(x)
        K = np.ones((N,N))
        
        for i in range(N):
            for j in range(i+1, N):
                cov = kernel(x[i], x[j])
                K[i][j] = cov
                K[j][i] = cov
                
        K = K + R *np.eye(N)
    
        return K
    
    def calculate_Ks(self, x):
        K_star =np.zeros((self.N,1))
        
        for i in range(self.N):
            K_star[i] = self.kernel(self.x[i], x)
            
        return K_star
    
    def setup_K(self):
        self.K = self.calculate_K(self.x, self.kernel, self.R)
    
    def predict(self,x):
        cov = 1 + self.R*self.kernel(x,x)
        
        K_star = self.calculate_Ks(x)
            
        m_expt = (K_star.T * np.mat(self.K).I) * np.mat(self.y).T
        
        K_expt = cov + self.R - (K_star.T * np.mat(self.K).I)*K_star
        
        return m_expt, K_expt
    
    @staticmethod
    def get_probability_old(K, y, R):
        multiplier = np.power(np.linalg.det(2 * np.pi * K), -0.5)
        return multiplier * np.exp((-0.5) * (np.mat(y) * np.dot(np.mat(K).I, y).T))
    
    @staticmethod
    def get_probability(K, y, R):    
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        logp = (
            -0.5 * np.dot(y.T, alpha)
            - np.sum(np.log(np.diag(L)))
            - K.shape[0] * 0.5 * np.log(2 * np.pi)
        )
        return logp

#    def optimize(self, R_list, L_list):
#        def kernel_proxy(length,f):
#            def wrapper(*args, **kwargs):
#                kwargs.update({"length": length})
#                return f(*args, **kwargs)
#            return wrapper
#        
#        history = []
#        
#        for r in R_list:
#            best_beta = (0,0)
#            for length in L_list:
#                K = self.calculate_K(self.x, kernel_proxy(length, self.kernel),r)
#                marginal = length*float(self.get_probability(K, self.y, r))
#                
#                if marginal > best_beta[0]:
#                    best_beta = (marginal,length)
#                    
#            history.append((best_beta[0], r, best_beta[1]))
#            
#        return sorted(history)[-1], np.mat(history)
    
    def optimize(self, *args, **kwargs):
        
        def kernel_proxy(f, *args):
            length = args[0]
            const = args[-1]
            
            if f == self.kernel_periodic:
                period = args[1]
                
            def wrapper(*args, **kwargs):
                kwargs.update({"length": length})
                kwargs.update({"const": const})
                if f == self.kernel_periodic or f == self.mix1:
                    kwargs.update({"period": period})
                return f(*args, **kwargs)
            return wrapper
        
        
        
        if len(args) == 2:
            R_list = args[0]
            L_list = args[1]
            
            landscape = np.zeros((len(R_list), len(L_list)))
            
            for i, r in enumerate(tqdm(R_list)):
                for j, l in enumerate(L_list):
                    K = self.calculate_K(self.x, kernel_proxy(self.kernel,l),r)
                    landscape[i,j]= self.get_probability(K, self.y, r)
                    
            index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)
            
            best_params = [R_list[index[0]], L_list[index[1]]]
            
        elif len(args)==3:
            R_list = args[0]
            L_list = args[1]
            P_list = args[2]
            
            landscape = np.zeros((len(R_list), len(L_list), len(P_list)))
            
            for i, r in enumerate(tqdm(R_list)):
                for j, l in enumerate(L_list):
                    for k, p in enumerate(P_list):
                        K = self.calculate_K(self.x, kernel_proxy(self.kernel, l, p),r)
                        landscape[i,j,k]= self.get_probability(K, self.y, r)
                    
            index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)
            best_params = [R_list[index[0]], L_list[index[1]], P_list[index[2]]]
        
                
        return landscape, best_params
    
    
    def optimize3(self, R_list, L_list):
        def kernel_proxy(length,f):
            def wrapper(*args, **kwargs):
                kwargs.update({"length": length})
                return f(*args, **kwargs)
            return wrapper
        
        landscape = np.zeros((len(R_list), len(L_list)))
        
        for i, r in enumerate(tqdm(R_list)):
            for j, l in enumerate(L_list):
                K = self.calculate_K(self.x, kernel_proxy(l, self.kernel),r)
                landscape[i,j]= self.get_probability(K, self.y, r)
                
        index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)
        
        return landscape, R_list[index[0]], L_list[index[1]]
    
    
    def optimizer(self, *args, **kwargs):
        
        def kernel_proxy(f, *args):
            length = args[0]
            const = args[-1]
            
            if f == self.kernel_periodic:
                period = args[1]
                
            def wrapper(*args, **kwargs):
                kwargs.update({"length": length})
                kwargs.update({"const": const})
                if f == self.kernel_periodic or f == self.mix1:
                    kwargs.update({"period": period})
                return f(*args, **kwargs)
            return wrapper
        
        lists = args
        
        n_elements = len(list(product(*lists)))
        
        landscape = np.zeros(n_elements)
        
        for i, item in enumerate(tqdm(list(product(*lists)))):
            K = self.calculate_K(self.x, kernel_proxy(self.kernel,*item[1:]),item[0])
            landscape[i]= self.get_probability(K, self.y, item[0])
            
        forma = []
        for lista in lists:
            forma.append(len(lista))
        
        landscape.shape = tuple(forma)
        
        index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)
        
        best_params = []
        for i,lista in enumerate(lists):
            best_params.append(lista[index[i]])
                        
        return landscape, best_params
        
#%%
def f1(x):
    return np.cos(.7*x).flatten()
        
def create_case(x, x_guess, y, kernel, R=0, title = False, save = False, orig_function = False, f = f1):
    gaus = GPR(x, y, kernel, R=R)
    y_pred = np.vectorize(gaus.predict)(x_guess)
    
    ax = plt.gca()
    plot_mu, = ax.plot(x_guess, y_pred[0], c="b")
    plt.gca().fill_between(x_guess, y_pred[0]-np.sqrt(y_pred[1]), y_pred[0]+np.sqrt(y_pred[1]), color="lightsteelblue")
    if title != False:
        plt.title(title)
    plot_misure = ax.scatter(x, y, c="black")
    plt.xlabel("x")
    plt.ylabel("y")
    legend_elements = [plot_mu, plot_misure]
    legend_labels =  ["media processo", "misure"]
    if orig_function != False:
        cosine, = ax.plot(x_guess, f(x_guess))
        legend_elements.append(cosine)
        legend_labels.append("f(x)")
    plt.legend(legend_elements, legend_labels, loc=2)
    if save != False:
        plt.savefig(save+".png", bbox_inches='tight')
    
    
def prior(x, x_guess,kernel, R=0):
    n = len(x_guess)
    Kss = GPR.calculate_K(x_guess, kernel, R=0)
    Lss = np.linalg.cholesky(Kss + 1e-6*np.eye(n))
    f_prior = np.dot(Lss, np.random.normal(size=(n,10)))
    
    plt.figure()
    plt.clf()
    plt.plot(x_guess, f_prior)
    plt.title("10 estrazioni dalla distribuzione a priori")
    plt.xlim([x_guess.min(), x_guess.max()])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('prior.png', bbox_inches='tight')
    
def post(x, x_guess, y,kernel, R=0, plot_mu = False):
    N = len(x)
    n = len(x_guess)
    
    n_draws = 10
    gaus = GPR(x, y, kernel, R=R)
    y_pred = np.vectorize(gaus.predict)(x_guess)
    
    K = np.mat(gaus.K) + 1e-6*np.eye(N)
    L = np.linalg.cholesky(K)

    
    Kss = GPR.calculate_K(x_guess, kernel, R=0)

    Ks = np.zeros((n,N))
    for i in range(len(x_guess)):
        Ks[i] = np.squeeze(gaus.calculate_Ks(x_guess[i]))
        
    Lk = np.linalg.solve(L, Ks.T)
    L2 = np.linalg.cholesky(Kss + 1e-6 * np.eye(n) - np.dot(Lk.T, Lk))
    
    # f_post = mu + L*N(0,1)
    f_post = np.tile(y_pred[0], (n_draws,1)).T + np.dot(L2, np.random.normal(size=(n, n_draws)))
    
    plt.figure()
    plt.clf()
    ax = plt.gca()
    if plot_mu == True:
        plot_mu,= ax.plot(x_guess, y_pred[0], c="b")
        plt.gca().fill_between(x_guess, y_pred[0]-np.sqrt(y_pred[1]), y_pred[0]+np.sqrt(y_pred[1]), color="lightsteelblue")
    ax.plot(x_guess, f_post)
    
    plt.title("10 estrazioni dalla distribuzione predittiva")
    plt.xlabel("x")
    plt.ylabel("y")
    plot_misure = ax.scatter(x, y, c="black")
    plt.legend([plot_mu, plot_misure], ["media processo", "misure"])
    
    plt.savefig('post.png', bbox_inches='tight')

def clr():
    return plt.close('all')



      