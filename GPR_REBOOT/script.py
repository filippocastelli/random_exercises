from GP_REBOOT import GPR_reboot
import numpy as np
import matplotlib.pylab as plt


def f1(x):
    return np.sin(x.T)
#%% IMPOSTAZIONE PROBLEMA

#f = lambda x: np.cos(.7*x).flatten()

N = 5     # numero punti training
n = 1000   # numero punti test
s = 0.0   # noise variance

rng = np.random.RandomState(2)
x = np.squeeze(rng.uniform(-5, 5, size = (N,1)))
x_guess = np.linspace(-5, 5, n)
y = f1(x) + s*np.random.randn(N)
#%%
# PLOT MISURE
plt.figure()
plt.title("Misure")
ax = plt.gca()
cosine, = ax.plot(x_guess, f1(x_guess))
measures = plt.scatter(x,y, c = "black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend([cosine, measures], ["f(x)", "punti training"])
plt.savefig('misure.png', bbox_inches='tight')

#%%
gaus = GPR_reboot(x,y,
                x_guess,
                kernel=GPR_reboot.gaussian_kernel,
                kernel_params = {'const': 1,
                                 'length': 1},
                R =0.)
#%%
#predictor
preds = gaus.predict()

gaus.plot(axlabels = ["x", "y"],title = 'before_optim', save = "test_figure")

#%%
optimizer_ranges = {'const': (0,10),
                    'length': (0,10)}
#
optimized_params, logp, grid, grid_values = gaus.optimizer(ranges_dict = optimizer_ranges,
                                                           mode = "asd",
                                                           Ns = 500,
                                                           output_grid = True)

gaus.update_params(optimized_params)
preds2 = gaus.predict()
gaus.plot(title = "after optim",axlabels = ["x", "y"], save = "optim")

