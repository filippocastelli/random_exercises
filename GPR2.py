import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

#%%

class GPR(object):
    
    #TODO: add kernels
    
    @classmethod #maybe update params
    def kernel_bell_shape(cls, x, y, delta=1.0):
        return np.exp(-0.5 * np.power(x-y, 2) / delta)
    
    @classmethod
    def kernel_laplacian(cls, x,y,delta=1):
        return np.exp(-0.5*np.abs(x-y) / delta)
    
    @classmethod #maybe update params
    def generate_kernel(cls, kernel, delta=1):
        def wrapper(*args, **kwargs):
            kwargs.update({"delta": delta})
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
        self.kernel = kernel if kernel else self.kernel_bell_shape
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
    def get_probability(K, y, R):
        multiplier = np.power(np.linalg.det(2 * np.pi * K), -0.5)
        return multiplier * np.exp((-0.5) * (np.mat(y) * np.dot(np.mat(K).I, y).T))
        
    def optimize(self, R_list, B_list):
        def kernel_proxy(delta,f):
            def wrapper(*args, **kwargs):
                kwargs.update({"delta": delta})
                return f(*args, **kwargs)
            return wrapper
        
        history = []
        
        for r in R_list:
            best_beta = (0,0)
            for b in B_list:
                K = gaus.calculate_K(self.x, kernel_proxy(b, self.kernel),r)
                marginal = b*float(self.get_probability(K, self.y, r))
                
                if marginal > best_beta[0]:
                    best_beta = (marginal,b)
                    
            history.append((best_beta[0], r, best_beta[1]))
            
        return sorted(history)[-1], np.mat(history)
#%%
        
def create_case(kernel, R=0):
    gaus = GPR(x, y, kernel, R=R)
    y_pred = np.vectorize(gaus.predict)(x_guess)
    
    plt.plot(x_guess, y_pred[0], c="b")
    plt.gca().fill_between(x_guess, y_pred[0]-np.sqrt(y_pred[1]), y_pred[0]+np.sqrt(y_pred[1]), color="lightsteelblue")
    plt.scatter(x, y, c="black")
    
    
def prior(kernel, R=0):
    Kss = GPR.calculate_K(x_guess, kernel, R=0)
    Lss = np.linalg.cholesky(Kss + 1e-6*np.eye(n))
    f_prior = np.dot(Lss, np.random.normal(size=(n,10)))
    
    plt.plot(x_guess, f_prior)
    
def post(kernel, R=0):
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
    f_post = np.tile(y_pred[0], (5,1)).T + np.dot(L2, np.random.normal(size=(n, 5)))
    
    plt.plot(x_guess, y_pred[0], c="b")
    plt.gca().fill_between(x_guess, y_pred[0]-np.sqrt(y_pred[1]), y_pred[0]+np.sqrt(y_pred[1]), color="lightsteelblue")
    plt.plot(x_guess, f_post)
    plt.scatter(x, y, c="black")

      
#%% IMPOSTAZIONE PROBLEMA
    
f = lambda x: np.cos(.7*x).flatten()

N = 5     # numero punti training
n = 500   # numero punti test
s = 0.    # noise variance

rng = np.random.RandomState(2)
x = rng.uniform(-5, 5, size = (N,1))
x_guess = np.linspace(-5, 5, n)
y = f(x) + s*np.random.randn(N)
#%% REGRESSORE 1
gaus = GPR(x,y)
y_pred = np.vectorize(gaus.predict)(x_guess)

plt.scatter(x,y, c="black")
plt.plot(x_guess, y_pred[0], c="b")
plt.gca().fill_between(x_guess, y_pred[0]-np.sqrt(y_pred[1]), y_pred[0]+np.sqrt(y_pred[1]), color="lightsteelblue")
    
#%% EFFETTI TERMINE RUMORE
plt.figure(figsize=(16, 16))
for i, r in enumerate([0.0001, 0.03, 0.09, 0.8, 1.5, 5.0]):
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, delta={}, beta={}".format("bell shape", 1, r))
    create_case(
        GPR.generate_kernel(GPR.kernel_bell_shape, delta=1), R=r)
    
#%% EFFETTI TERMINE LUNGHEZZA
plt.figure(figsize=(16, 16))
for i, d in enumerate([0.05, 0.5, 1, 3.2, 5.0, 7.0]):
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, delta={}, beta={}".format("kernel_laplacian", d, 1))
    create_case(
        GPR.generate_kernel(GPR.kernel_bell_shape, delta=d), R=0)

#%%

gaus = GPR(x, y)
R_list = np.linspace(0.0, 1, 100)
B_list = np.linspace(0.1, 10, 20)
best_params, history = gaus.optimize(R_list, B_list)

plt.figure()
plt.plot(history[:,1], history[:,2])
print("best parameters (probability, r, b): ", best_params)
plt.show()

plt.figure()
create_case(GPR.generate_kernel(GPR.kernel_bell_shape, delta=best_params[2]), R=best_params[1])



#%%
post(GPR.kernel_bell_shape)
