# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>s
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data, modify_legend, predict_plot

def f(x):
    """The function to predict."""
    return x * np.sin(x)


#%%
# INTERPOLATION CASE
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
x_guess = x
#%%
#SKLEARN GPR REGRESSOR
kernel_sklearn = C(4.71, (1e-3, 1e3)) * RBF(1.68, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel_sklearn, n_restarts_optimizer=0)
gp.fit(X, y)
y_pred, sigma = gp.predict(x, return_std=True)

plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
#%%
# MY GPR REGRESSOR
my_kernel = GPR.generate_kernel(GPR.kernel_gaussian,
                                const = 22.19 ,
                                length = 1.78 )

N = 10
n = 20
s = 0

rng = np.random.RandomState(2)
my_x = rng.uniform(-5, 5, size = (N,1))
x_guess = np.linspace(-5, 5, n)
my_y = f1(x)


my_y = np.squeeze(f(X))
plt.figure()
create_case(x = X,
            y = my_y,
            x_guess = np.squeeze(x_guess),
            kernel = my_kernel,
            R = 1e-6,
            title = "my_gpr_1",
            save = False,
            )


gaus = GPR(x,y)
paramlist = {'length': np.linspace(0.1,100,100),
             'const': np.linspace(0.1, 10, 10)}

landscape, best_params = gaus.optimizer(param_dictionary=paramlist, noiselist = False, parallel = False)


#plt.plot(x, f(x))
#%%# ----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()