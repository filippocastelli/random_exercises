from GP_REBOOT import GPR_reboot
import numpy as np
import matplotlib.pylab as plt

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def f(x):
    """The function to predict."""
    return x * np.sin(x)

#%%
# INTERPOLATION CASE
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
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
ax = plt.gca()

params = {'const': np.exp(4.71),
          'length': np.exp(1.68)}


gaus = GPR_reboot(x = X, y =y,
                  x_guess = x,
                  kernel = GPR_reboot.gaussian_kernel,
                  kernel_params = params,
                  R = 0.)

gaus.predict()
gaus.plot(ax = ax)
