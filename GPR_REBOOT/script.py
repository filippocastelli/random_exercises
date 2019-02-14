from GP_REBOOT import GPR_reboot
import numpy as np
import matplotlib.pylab as plt


def f1(x):
    return np.sin(x.T)
#%% IMPOSTAZIONE PROBLEMA

#f = lambda x: np.cos(.7*x).flatten()

N = 5     # numero punti training
n = 100   # numero punti test
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

preds = gaus.predict()

asse = gaus.plot(axlabels = ["x", "y"], save = "test_figure")
