import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm

import warnings 
sns.set(color_codes=True)
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import pickle
from scipy.optimize import minimize as fmin
from scipy.optimize import brute as brute_optim


#%%


class GPR_reboot(object):
    
# =============================================================================
#     GPR: GAUSSIAN PROCESS REGRESSOR
#     main class
#     
#     by Filippo Maria Castelli
#     last major update 13/02/2019
# =============================================================================
    
# =============================================================================
#     INIT
# =============================================================================
    def __init__(self,x,y,x_guess,kernel = False,kernel_params = False,R = 0):
        
        self.x = np.squeeze(x)
        self.y = np.squeeze(y)
        self.x_guess = np.squeeze(x_guess)
        self.N = len(self.x)
        self.R = np.squeeze(R)
        
        self.kernel = kernel if kernel else self.gaussian_kernel
        
        self.params = kernel_params
        if kernel_params == {}:
            warnings.warn("params for {} kernel not set!, remember to set them before using the predict method".format(kernel)) 
        else:
            self.calc_kernel_matrices()
            self.kernel_setup()
            
        self.kernel_arguments = self.find_arguments(self.kernel)
# =============================================================================
#  KERNELS:
#        # SHARED METHODS:
#            classmethods:
#                > wrap_kernel(cls, kernel, **kwargs)
#            staticmethods:
#                > find_arguments(kernel)
#                > difference_mat(data1, data2)
#            instancemethods:
#                > kernel_setup(self)
#        # KERNEL FUNCTIONS:  
#            classmethods:
#                > gaussian_kernel(cls, data1, data2, length, const)
#                > periodic_kernel(cls, data1, data2, const, period, length)
#                > rational_quadratic(cls, data1, data2, const, length, alpha)
# =============================================================================
        
# =============================================================================
#     # > SHARED METHODS FOR KERNEL EVALUATION
# =============================================================================
    # > KERNEL FUNCTION WRAPPING
    @classmethod
    def wrap_kernel(cls, kernel, **kwargs):
        
        arguments = cls.find_arguments(kernel)
        argument_dict = kwargs
        arglist = tuple(argument_dict.keys())
        
        assert len(arguments) == len(arglist), "wrong number of arguments for kernel!"
        assert not sum([not i in arguments for i in arglist]), "wrong arguments have been passed to the wrapper"
        
        
        def wrapper(*args, **kwargs):
            for param, paramvalue in argument_dict.items():
                kwargs.update({param: paramvalue})
                
            #FUTURE: handle gradient optimization
            #when gradient optimization is implemented
            #you may want to add a differente update for gradient flag
            #kwargs.update({"wantgrad": wantgrad})
            
            return kernel(*args, **kwargs)
        return wrapper
        
    # > FIND KERNEL ARGUMENTS
    @staticmethod
    def find_arguments(kernel):
        varnames = kernel.__code__.co_varnames

        try:
            kernel_arguments = varnames[3: varnames.index('const') +1]
        except ValueError:
            raise Exception(f"kernel function {kernel} not valid")

        return kernel_arguments
    
    # > MATRIX DIFFERENCE BETWEEN VECTORS
    @staticmethod
    def difference_mat(data1, data2):
        
        dim1 = len(data1)
        dim2 = len(data2)
        
        dvec1 = np.tile(data1, (dim2,1))
        dvec2 = np.tile(data2, (dim1,1)).T

        diff = (dvec1 - dvec2)
        
        return diff
    
    # > KERNEL SETUP
    def kernel_setup(self):
        assert self.params != {}, "Kernel parameters not set!"
        self.wrapped_kernel = self.wrap_kernel(self.kernel, **self.params)
        
    def update_params(self, newparams_dict):
        newparams_names = tuple(newparams_dict.keys())
        assert len(self.kernel_arguments) == len(newparams_names), "wrong number of parameters for kernel!"
        assert not sum([not i in newparams_names for i in self.kernel_arguments]), "you're trying to update a different list of parameters"
        
        self.params = newparams_dict
        self.kernel_setup()
        self.calc_kernel_matrices()
        
# =============================================================================
#     KERNEL FUNCTIONS
#        a standard kernel function should input data1, data2 and parameters
#        remember ALWAYS to make const the last parameter as it's position
#        is needed when wrapping and passing arguments
#        
#        FUTURE:
#            when grad optimization is implemented, the grad flag will be the
#            last argument
# =============================================================================
    # > GAUSSIAN KERNEL
    @classmethod
    def gaussian_kernel(cls,
                        data1,
                        data2,
                        length = 1,
                        const = 1):
    
        square_diff = cls.difference_mat(data1, data2)**2
        
        k = np.square(const)*np.exp(-2*(square_diff/np.square(length)))
        
        return k
    
    # > PERIODIC KERNEL
    @classmethod
    def periodic_kernel(cls,
                        data1,
                        data2,
                        period = 1,
                        length = 1,
                        const = 1):
        
        abs_diff = np.abs(cls.difference_mat(data1, data2))
        
        
        k = np.square(const) * np.exp(-2 * np.square(np.sin((np.pi/period)*abs_diff/np.square(length))))
        
        return k
    
    
    # > RATIONAL QUADRATIC KERNEL
    @classmethod
    def rational_quadratic(cls,
                           data1,
                           data2,
                           alpha = 1,
                           length = 1,
                           const = 1):
        
        squared_diff = cls.difference_mat(data1, data2)**2
        
        
        k = np.square(const) * np.power(( 1 + squared_diff  / (2*alpha*np.square(length))), -alpha)
        
        return k
    
# =============================================================================
# K, K*, K** CALCULATIONS
#    instancemethods:
#        > calc_K(self)
#        > calc_Ks(self)
#        > calc_Ks(self)
#        > calc_kernel_matrices(self)
# =============================================================================
     
    # TODO: add exceptions
    # I may need to add some exceptions for the case where no params are ready
    
    # NOTE: the only reason why there are three separate methods for doing
    # basically the same thing is to avoid argument confusion
    
    # TODO: reconsider if this kind of architecture for matrix calculation makes sense
    # to calculate K, K* and K** outside you shoud first wrap a kernel and 
    # then use it with (x,x) , (x,y) and (y,y). I may change this architecture
    # in the future.
    
    def calc_K(self):
        return self.wrapped_kernel(self.x,self.x)
    
    def calc_Ks(self):
        return self.wrapped_kernel(self.x_guess, self.x)

    def calc_Kss(self):
        return self.wrapped_kernel(self.x_guess, self.x_guess)
        
    def calc_kernel_matrices(self):
        print(">calculating K, Ks, Kss...")
        self.kernel_setup()
        self.K = self.calc_K()
        self.Ks = self.calc_Ks()
        self.Kss = self.calc_Kss()
        
# =============================================================================
# PREDICTIONS
#        staticmethods
#            >calc_logp(alpha, L, y)ù
#            > get_L_alpha(K,y)
#        instancemethods
#            >calc_logp(self)
# =============================================================================
    @staticmethod
    def calc_logp(alpha, L, y):
        logp = -0.5*np.dot(y.T,alpha) - np.trace(L) - 0.5*len(y)*np.log(2*np.pi)
        return logp
        
    @staticmethod
    def get_L_alpha(K,y):
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        return L, alpha
    
    def predict(self):
        
        K_noise = self.K + self.R*np.eye(self.N)
        
        L, alpha = self.get_L_alpha(K_noise, self.y)
        
        self.logp = self.calc_logp(alpha, L, self.y)
        
        self.pred_y = np.dot(self.Ks.T, alpha)
        
        v = np.linalg.solve(L, self.Ks)
        self.pred_variance = np.diag(np.diag(self.Kss) - np.dot(v.T, v))
        
        return self.pred_y, self.pred_variance, self.logp
        
    
# =============================================================================
# PLOTS
#        staticmethods
#            >create_figure(title, axlabels)
#            >save_figure(ax,title)
#        instancemethods
#            >plot_process(self, mean, var, x_guess, ax)
#            >plot_measures(self,x,y,ax)
#            >plot(self, plot_process, plot_measures, title, save, return_ax
#                  x,y,x_guess, pred_y, var_pred)
# =============================================================================
    
    
    @staticmethod
    def create_figure(title, axlabels = None):
        fig, ax = plt.subplots()
        ax.set_title(title)
        if axlabels is not None:
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[1])
        return ax
    
    @staticmethod
    def save_figure(ax, title):
        plt.sca(ax)
        plt.savefig(title)

    def plot_process(self,mean, var, x_guess, ax):
        
        std_dev = np.sqrt(np.sqrt(var**2))
        ax.plot(x_guess, mean, label = 'media_processo')
        ax.fill_between(x_guess,
                        mean - std_dev,
                        mean + std_dev,
                        color = "lightsteelblue",
                        label = 'std_processo')
        
    def plot_measures(self,x, y,ax):
        ax.scatter(x, y, label = 'misure')
        
    def plot(self,
             plot_process = True,
             plot_measures = True,
             title = "Gaussian Process Regression",
             axlabels = None,
             save = False,
             return_ax = False,
             x = None,
             y = None,
             x_guess = None,
             pred_y = None,
             var_pred = None):
        
        pred_y = pred_y if pred_y is not None else self.pred_y
        var_pred = var_pred if var_pred is not None else self.pred_variance
        x_guess = x_guess if x_guess is not None else self.x_guess
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        
        ax = self.create_figure(title, axlabels)
        
        if plot_process:
            self.plot_process(mean = pred_y,
                              var = var_pred,
                              x_guess = x_guess,
                              ax = ax)
        if plot_measures:
            self.plot_measures(x = x,
                               y = y,
                               ax = ax)
            
        if save != False:
            self.save_figure(ax, save)
            
            
        ax.legend()
        if return_ax:
            return ax
        
# =============================================================================
#  OPTIMIZATION       
#            instancemethods
#                > optimizer(self, ranges_dict, Ns, output_grid)
# =============================================================================

    # > OPTIMIZER
    # at the moment is a simple brute force optimizer
    
    #TODO: implement a gradient optimizer
    def optimizer(self,
                  mode = 'brute',
                  ranges_dict = None,
                  Ns = 100,
                  output_grid = False):
        
        param_names = tuple(ranges_dict.keys())
        param_ranges = tuple(ranges_dict.values())
        
        modes = ['brute']
        assert mode in modes, "please select a valid mode for the optimizer, choose between: {}".format(*modes)
        returns = []


        
        #NOT THE MOST EFFICIENT THING EVER
        def logp(x, *args):
            params = x
            param_names = args
            
            param_dict = dict(zip(param_names, params))
            w_kernel = self.wrap_kernel(self.kernel, **param_dict)
            try:
                K = w_kernel(self.x, self.x)
                L, alpha = self.get_L_alpha(K,self.y)
                logp = self.calc_logp(alpha, L , self.y)
            except np.linalg.LinAlgError: 
                logp = -np.inf
            
            
            return -logp
            
        
        if mode == 'brute':
            x0, fval, grid, Jout = brute_optim(func = logp,
                                               ranges = param_ranges,
                                               args = (param_names),
                                               Ns = Ns,
                                               full_output = True,
                                               disp = True)
        

            optim_params = dict(zip(param_names, x0))
            returns.append(optim_params)
            returns.append(-fval)
            
            if output_grid:
                returns.append(grid)
                returns.append(Jout)
            
            
        return returns
        
        
        