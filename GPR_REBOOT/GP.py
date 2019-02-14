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


#%%


class GPR(object):
    
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
    def __init__(self,x,y,x_guess,kernel = None,kernel_params = {},R = 0):
        
        #        super().__init__()
        self.x = np.squeeze(x),
        self.y = np.squeeze(y),
        self.x_guess = np.squeeze(x_guess),
        self.N = len(self.x),
        self.R = np.squeeze(R),
        
        self.kernel = kernel if kernel else self.kernel_gaussian
        
        self.params = kernel_params
        if kernel_params == {}:
            warnings.warn("params for {} kernel not set!, remember to set them before using the predict method".format(kernel)) 
        else:
            self.calc_kernel_matrices()
            self.kernel_setup()

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
        return self.wrapped_kernel(self.x, self.x_guess)

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
# =============================================================================
    @staticmethod
    def calc_logp(alpha, L, y):
        logp = -0.5*np.dot(y.T,alpha) - np.trace(L) - 0.5*len(y)*np.log(2*np.pi)
        return logp
        
    def predict(self):
        
        K_noise = self.K + self.R*np.eye(self.N)
        
        L = np.linalg.cholesky(K_noise)
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        
        self.logp = self.calc_logp(alpha, L, self.y)
        
        self.pred_y = np.dot(self.Ks.T, alpha)
        
        v = np.linalg.solve(L, self.Ks)
        self.pred_variance = np.diag(self.Kss) - np.dot(v.T, v)
        
        return self.y_pred, self.pred_variance, self.logp
        
    
        
        
        
        
        
        