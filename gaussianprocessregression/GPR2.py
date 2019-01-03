import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

#%%

class GPR(object):
    
    #TODO: add kernels
    
    @classmethod #maybe update params
    def kernel_gaussian(cls, x, y, length=1.0):
        return np.exp(-0.5 * np.power(x-y, 2) / length)
    
    @classmethod
    def kernel_laplacian(cls, x,y,length=1):
        return np.exp(-0.5*np.abs(x-y) / length)
    
    @classmethod #maybe update params
    def generate_kernel(cls, kernel, length=1):
        def wrapper(*args, **kwargs):
            kwargs.update({"length": length})
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

    def optimize(self, R_list, L_list):
        def kernel_proxy(length,f):
            def wrapper(*args, **kwargs):
                kwargs.update({"length": length})
                return f(*args, **kwargs)
            return wrapper
        
        history = []
        
        for r in R_list:
            best_beta = (0,0)
            for length in L_list:
                K = gaus.calculate_K(self.x, kernel_proxy(length, self.kernel),r)
                marginal = length*float(self.get_probability(K, self.y, r))
                
                if marginal > best_beta[0]:
                    best_beta = (marginal,length)
                    
            history.append((best_beta[0], r, best_beta[1]))
            
        return sorted(history)[-1], np.mat(history)
    
    def optimize2(self, R_list, L_list):
        def kernel_proxy(length,f):
            def wrapper(*args, **kwargs):
                kwargs.update({"length": length})
                return f(*args, **kwargs)
            return wrapper
        
        landscape = np.zeros((len(R_list), len(L_list)))
        
        for i, r in enumerate(R_list):
            for j, l in enumerate(L_list):
                K = gaus.calculate_K(self.x, kernel_proxy(l, self.kernel),r)
                landscape[i,j]= self.get_probability(K, self.y, r)
                
        index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)
        
        
                
        return landscape, R_list[index[0]], L_list[index[1]]
        
#%%
        
def create_case(kernel, R=0, title = False, save = False, orig_function = False):
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
    
    
def prior(kernel, R=0):
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
    
def post(kernel, R=0, plot_mu = False):
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
      
#%% IMPOSTAZIONE PROBLEMA
f = lambda x: np.cos(.7*x).flatten()

N = 5     # numero punti training
n = 500   # numero punti test
s = 0.001   # noise variance

rng = np.random.RandomState(2)
x = rng.uniform(-5, 5, size = (N,1))
x_guess = np.linspace(-5, 5, n)
y = f(x) + s*np.random.randn(N)
#%%
# PLOT MISURE
plt.figure()
plt.title("Misure")
ax = plt.gca()
cosine, = ax.plot(x_guess, f(x_guess))
measures = plt.scatter(x,y, c = "black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend([cosine, measures], ["f(x)", "punti training"])
plt.savefig('misure.png', bbox_inches='tight')
#%%
# PLOT PRIORI
prior(GPR.kernel_gaussian)
#%%
# PLOT POSTERIORI
post(GPR.kernel_gaussian)
#%% REGRESSORE 1
plt.figure()
ax = plt.gca()
create_case(GPR.kernel_gaussian, title = "GPR", save = "GPR", orig_function = True)
cosine = ax.plot(x_guess, f(x_guess), c="red")
#%% EFFETTI TERMINE RUMORE
plt.figure(figsize=(16, 16))
noiseplot_length = 3
for i, r in enumerate([0.0001, 0.03, 0.09, 0.8, 1.5, 5.0]):
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", noiseplot_length, r))
    create_case(GPR.generate_kernel(GPR.kernel_gaussian, length=noiseplot_length), R=r)
plt.savefig('noise_plot.png', bbox_inches='tight')
#%% EFFETTI TERMINE LUNGHEZZA
plt.figure(figsize=(16, 16))
lengthplot_noise = 1e-3
for i, l in enumerate([0.05, 0.5, 1, 3.2, 5.0, 7.0]):   
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", l, lengthplot_noise))
    create_case(
        GPR.generate_kernel(GPR.kernel_gaussian, length=l), R=lengthplot_noise)
plt.savefig('length_plot.png', bbox_inches='tight')
#%%
gaus = GPR(x, y)
R_list = np.linspace(0.01, 1, 100)
L_list = np.linspace(0.1, 100, 100)
best_params, history = gaus.optimize(R_list, L_list)

plt.figure()
plt.title("Prob. history")
plt.plot(history[:,1], history[:,2])
print("best parameters (probability, r, b): ", best_params)
plt.show()

plt.figure()
create_case(GPR.generate_kernel(GPR.kernel_gaussian, length=best_params[2]), R=best_params[1], title = "Parametri Otimizzati")
plt.savefig('optimal_params.png', bbox_inches='tight')
#%%

gaus = GPR(x, y)
R_list = np.linspace(0.0001, 0.1, 100)
L_list = np.linspace(0.1, 100, 100)
lml, r_opt, l_opt= gaus.optimize2(R_list, L_list)

#plt.figure()
#plt.title("Prob. history")
#plt.plot(history[:,1], history[:,2])
#print("best parameters (probability, r, b): ", best_params)
#plt.show()

plt.figure()
plt.imshow(lml, cmap = 'viridis')
plt.figure()
create_case(GPR.generate_kernel(GPR.kernel_gaussian, length=l_opt), R=r_opt, title = "Parametri Otimizzati")
plt.savefig('optimal_params.png', bbox_inches='tight')