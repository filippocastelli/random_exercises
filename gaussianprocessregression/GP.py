import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set(color_codes = True)
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import pickle
from scipy.optimize import minimize as fmin
#%%

class GPR(object):

    #TODO: add kernels

    @classmethod
    def mix1(cls, x, y, length=1.0, length2=1.0, period= 1, const= 1.0, const2 = 1.0):
        return cls.kernel_gaussian(x,y,length, const) + const2*cls.kernel_periodic(x,y,length2,period)

    @classmethod
    def kernel_gaussian(cls, x, y, length=1.0, const = 1.0, wantgrad=False):

        sq_dist = np.power(x-y,2)
        exponential = np.exp(-0.5 * sq_dist / length**2)
        k = (const**2)*exponential
        if wantgrad == False:
            return k
        else:
            gradient = np.zeros(2)
            gradient[0] = 2*const*exponential
            gradient[1] = (const**2)*exponential*sq_dist*np.power(length, -3)

            return gradient

    # TODO: fix derivatives and kernel
    @classmethod
    def kernel_periodic_decay(cls, x, y, length=1.0, const = 1.0, decay = 1.0, wantgrad=False):
        sq_dist = np.power(x-y,2)

        period = 1

        exp_arg_1 = -0.5 * sq_dist / decay**2

        squared_sin = np.power(np.sin(np.pi*(x-y)/period),2)
        exp_arg_2 = -2 * squared_sin/(length**2)

        exponential = np.exp(exp_arg_1 + exp_arg_2)

        k = (const**2)*exponential
        if wantgrad == False:
            return k
        else:
            gradient = np.zeros(3)
            gradient[0] = 2*const*exponential
            gradient[1] = (const**2)*exponential*sq_dist*np.power(decay, -3)
            gradient[2] = (const**2)*exponential*4*squared_sin*np.power(length, -3)

            return gradient

#    @classmethod
#    def kernel_laplacian(cls, x,y,length=1, const = 1.0):
#        return np.exp(-0.5*np.abs(x-y) / length)
#
#    # TODO: define gradient for periodic
#    @classmethod
#    def kernel_periodic_old(cls, x,y,length=1, period=1, const = 1.0, wantgrad = False):
#        sin_argument = np.pi*np.abs(x-y)/period
#        exp_argument = -2*np.power(np.sin(sin_argument),2)/length
#        return const*np.exp(exp_argument)

    @classmethod
    def kernel_periodic(cls, x,y,length = 1, period = 1,const = 1, wantgrad = False):
        #gradient is always given in [const, period, length] order
        
        sin_argument = np.pi * (x-y) / period
        exp_argument = -2 * np.power(np.sin(sin_argument), 2) / np.power(length, 2)
        
        k = np.power(const,2) * np.exp(exp_argument)
        
        if wantgrad == False:
            return k
        else:
            gradient = np.zeros(3)
            
            #dk/dc
            gradient[0] = 2*const*np.exp(exp_argument)
            #dk/dp
            gradpconst = (4* np.pi * const **2 )/ (length**2 * period**2 )
            gradient[1] = gradpconst * (x-y) * np.exp(exp_argument) * np.sin(sin_argument)*np.cos(sin_argument)
            
            #dk/dl
            gradlconst = 4*(const**2)/(length**2)
            
            gradient[2] = gradlconst * np.exp(exp_argument)*np.power(np.sin(sin_argument),2)
            
            return gradient
        

    @classmethod
    def kernel_rational_quadratic(cls, x, y, length=1.0, const = 1.0, shape = 1.0, wantgrad=False):
        sq_dist = np.power(x-y,2)

        argument = 1+ sq_dist/(2*shape*np.power(length,2))


        k = (const**2)*np.power(argument, -shape)

        if wantgrad == False:
            return k
        else:
            gradient = np.zeros(3)
            gradient[0] = 2*const*np.power(argument, -shape)
            gradient[1] = (const**2)*np.power(argument, -(shape+1))*sq_dist*np.power(length, -3)

            multiply_term = (sq_dist/(2*np.power(length,2)*shape))-np.log(argument)

            gradient[2] = (const**2)*np.power(argument, -shape)*multiply_term


            return gradient

    @classmethod
    def kernel_whitenoise(cls, x, y, const, wantgrad = False):

        def k_noise(x,y):
            if x == y:
                return np.power(const, 2)
            else:
                return 0

        k = k_noise(x,y)

        if wantgrad == False:
            return k
        else:
            gradient = np.zeros(1)

            gradient[0] = k

            return gradient



    @classmethod
    def find_arguments(cls,kernel):
        varnames = kernel.__code__.co_varnames
        try:
            kernel_arguments = varnames[varnames.index('y')+1:varnames.index('wantgrad')]
        except ValueError:
            raise Exception('kernel function {} not valid', kernel)

        return kernel_arguments


    @classmethod
    def generate_kernel(cls, kernel, wantgrad= False,**kwargs):

        kernel_arguments = cls.find_arguments(kernel)
        dict_arguments = kwargs
        arglist = tuple(kwargs.keys())

        assert len(kernel_arguments) == len(arglist)

        #checking if entries are the same
        assert not sum([not i in kernel_arguments for i in arglist])

        def wrapper(*args, **kwargs):
            for param, paramvalue in dict_arguments.items():
                kwargs.update({param: paramvalue})
            kwargs.update({'wantgrad': wantgrad})

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

    @classmethod
    def calculate_Kgrad(cls, x, gradkernel, R= 0):
        N = len(x)
        testgrad = gradkernel(x[0], x[0])
        theta_l = len(testgrad)

        Kgrad = np.ones((N,N,theta_l))

        for i in range(N):
            for j in range(i+1, N):
                grad = gradkernel(x[i], x[j])
                for k in range(theta_l):

                    Kgrad[i][j][k] = grad[k]
                    Kgrad[j][i][k] = grad[k]

        return Kgrad

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
    def get_probability(K, y, R=0):
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

    @staticmethod
    def grad_logp(K, gradK, y, R=0):
        try:
            L = np.linalg.cholesky(K)
            invk = np.linalg.solve(L.transpose(), np.linalg.solve(L, np.eye(len(y))))
        except np.linalg.LinAlgError:
            return 0

        ## TODO: maybe check if gradK and thetas are same length
        ntheta = gradK.shape[2]
        dlogp = np.zeros(ntheta)
        for i in range(ntheta):
            dlogp[i] = 0.5 * np.dot(y.T, np.dot(invk, np.dot(gradK[i], invk))) -0.5* np.trace(np.dot(invk, gradK[i]))

        return dlogp

    def calc_lml_element(self,params, keys):
        
        if keys[-1] == 'noise':
            paramkeys = keys[:-1]
            values_to_pass =  params[:-1]
            noise = params[-1]
        else:
            noise = 0
            
        params_to_pass = dict(zip(paramkeys, values_to_pass))
        
        ker = self.generate_kernel(self.kernel, **params_to_pass)
        K = self.calculate_K(self.x, ker, R = noise)
        return self.get_probability(K, self.y)

    @staticmethod
    def removekey(dictionary, key):
        r = dict(dictionary)
        del r[key]
        return r
    
    
    def optimizer(self,param_dictionary, noiselist = False, parallel = True):
        # TODO: check if grid optimizer still works
        # spoiler: it doesnt
        
        def product_dict(**kwargs):
            keys = kwargs.keys()
            vals = kwargs.values()
            for instance in product(*vals):
                yield dict(zip(keys, instance))
                
        def product_from_dict(**kwargs):
            vals = kwargs.values()
            
            for instance in product(*vals):
                yield instance
                
                
                
        if noiselist != False:
            param_dictionary['noise'] = noiselist
            
        cases_keys = tuple(param_dictionary.keys())
        cases_tuple = tuple(product_from_dict(**param_dictionary))
        all_possible_cases = tuple(product_dict(**param_dictionary))

        def kernel_proxy(f, *args):

            kernel_arguments = self.find_arguments(self.kernel)
            dict_arguments = args[0]
            arglist = tuple(dict_arguments.keys())

            assert len(kernel_arguments) == len(arglist)

            #checking if entries are the same
            assert not sum([not i in kernel_arguments for i in arglist])

            def wrapper(*args, **kwargs):
                for param, paramvalue in dict_arguments.items():
                    kwargs.update({param: paramvalue})
                return f(*args, **kwargs)
            return wrapper
        
        landscape = np.zeros(len(cases_tuple))


                
        if parallel == True:
            num_cores = multiprocessing.cpu_count()
            landscape_list = Parallel(n_jobs=num_cores)(delayed(self.calc_lml_element)(case,cases_keys) for case in tqdm(cases_tuple))
            landscape = np.array(landscape_list)

        else:
            #NON PARALLELIZED
            
            for i, case in enumerate(tqdm(all_possible_cases)):

                landscape[i] = self.calc_lml_element(case)
#                
##            for i, item in enumerate(tqdm(list(product(*lists)))):
##                K = self.calculate_K(self.x, kernel_proxy(self.kernel,*item[1:]),item[0])
##                landscape[i]= self.get_probability(K, self.y, item[0])
#                landscape[i] = self.calc_lml_element(kernel_proxy, i)

        forma = []
        for key, arg in param_dictionary.items():
            forma.append(len(arg))

        landscape.shape = tuple(forma)

        index = np.unravel_index(np.argmax(landscape, axis=None), landscape.shape)

        best_params = []
        
        for i, key in enumerate(param_dictionary.items()):
#        for i,lista in enumerate(lists):
            best_params.append(key[1][index[i]])

        return landscape, best_params

    def grad_optimizer(self, theta0):

        theta0_values = np.array(tuple(theta0.values()))
        theta0_names = tuple(theta0.keys())

        def wrapped_logp(thetavals, *args):
            params = {k:v for k, v in zip(theta0_names, thetavals)}

            kernel_theta = self.generate_kernel(self.kernel, **params)
            K = self.calculate_K(self.x, kernel_theta)
            logp = self.get_probability(K, self.y, R=0)


            return float(logp)

        def wrapped_gradlogp(thetavals, *args):
            params = {k:v for k, v in zip(theta0_names, thetavals)}

            kernel_theta = self.generate_kernel(self.kernel, **params)
            kernel_theta_grad = self.generate_kernel(self.kernel, wantgrad = True, **params)

            K = self.calculate_K(self.x, kernel_theta)
            Kgrad = self.calculate_Kgrad(self.x, kernel_theta_grad)

            dlogp = self.grad_logp(K, Kgrad, self.y)

            return np.array(dlogp)

        optim_theta = fmin(wrapped_logp, theta0_values, args=(), method='CG',
                                jac=wrapped_gradlogp,
                                tol=1e-6,
                                options={'disp': 1})

        return optim_theta
#%%
def f1(x):
    return np.cos(.7*x).flatten()

def create_case(x, x_guess, y, kernel, R=0,
                title = False,
                save = False,
                orig_function = False,
                load = False,
                f = f1):
    if load != False:
        f = open(load+'.gpr', 'rb')
        x_loaded, x_guess_loaded, y_loaded, R_loaded, y_pred_loaded= pickle.load(f)
        f.close()
        if (x == x_loaded).all() and (x_guess == x_guess_loaded).all() and (y_loaded == y).all() and (R_loaded == R):
            y_pred = y_pred_loaded
            gaus = GPR(x, y, kernel, R=R)
            print("file "+ load + ".gpr succesfully loaded")
        else:
            print("file "+ load + ".gpr can't be loaded")
            print("check if x, x_guess, y, y_pred are the same")
            raise Exception('cannot load {}.gpr file'.format(load))
    else:
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
        f = open(save+'.gpr', 'wb')
        pickle.dump([x, x_guess, y, R, y_pred], f)
        f.close()
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

def gen_data(x, x_guess, y,kernel, R=0,
             plot_mu = False, plot_variance = False,
             save = False, separate_figure = True):
    N = len(x)
    n = len(x_guess)

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

    f_post = np.zeros(n)
    for i, point in enumerate(x_guess):
        f_post[i] = y_pred[0][i] + np.dot(L2[:,i], np.random.normal(size = (n,1)))

    if separate_figure == True:
        plt.figure()
        plt.clf()

    ax = plt.gca()
    if plot_mu == True:
        plot_mu,= ax.plot(x_guess, y_pred[0], c="b")
    if plot_variance == True:
        plt.gca().fill_between(x_guess, y_pred[0]-np.sqrt(y_pred[1]), y_pred[0]+np.sqrt(y_pred[1]), color="lightgray")

    ax.scatter(x_guess, f_post, c = "red")

    plt.title("Dati generati dalla distribuzione predittiva")
    plt.xlabel("x")
    plt.ylabel("y")
#    plot_misure = ax.scatter(x, y, c="black")
    plot_dati = ax.scatter(x_guess, f_post, c = "red")
#    plt.legend([plot_mu,plot_dati, plot_misure], ["media processo","dati simulati", "misure"])
    plt.legend([plot_mu,plot_dati], ["media processo","dati simulati"])

    if save != False:
        plt.savefig(save+'.png', bbox_inches='tight')

