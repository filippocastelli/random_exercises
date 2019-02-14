import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data
from itertools import product
#%% IMPOSTAZIONE PROBLEMA

#f = lambda x: np.cos(.7*x).flatten()

N = 5     # numero punti training
n = 50   # numero punti test
s = 0.0   # noise variance

rng = np.random.RandomState(2)
x = rng.uniform(-5, 5, size = (N,1))
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

gen_data(x, x_guess,y, GPR.kernel_gaussian, R = 0.1, save = "gendata")
#%%
# PLOT PRIORI
kernel = GPR.generate_kernel(GPR.kernel_gaussian, const = 0.1, length = 5 )
prior(x = x,
      x_guess = x_guess,
      kernel = kernel, R =10)
#%%
# PLOT POSTERIORI
post(x = x,
     x_guess = x_guess,
     y= y,
     kernel = GPR.kernel_gaussian)
#%% REGRESSORE 1
plt.figure()
ax = plt.gca()
create_case(x = x,
            y= y,
            x_guess = x_guess,
            kernel = GPR.kernel_gaussian,
            R = 0.000,
            title = "GPR",
            save = "GPR",
            orig_function = f1)

cosine = ax.plot(x_guess, f1(x_guess), c="red")

#%% EFFETTI TERMINE RUMORE

N = 10 # numero punti training
n = 500   # numero punti test
s = 0.1   # noise variance

rng = np.random.RandomState(5)
x = rng.uniform(-5, 5, size = (N,1))
x_guess = np.linspace(-5, 5, n)
y = f1(x) +s*np.random.randn(N)
plt.figure(figsize=(16, 16))
noiseplot_length = 3

for i, r in enumerate([0.00001, 0.02, 0.1, 0.8, 1.5, 5.0]):
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", noiseplot_length, r))
    create_case(x,
                x_guess,
                y,
                GPR.generate_kernel(GPR.kernel_gaussian,
                                    const = 1,
                                    length=noiseplot_length),
                R=r)
    
plt.savefig('noise_plot.png',bbox_inches='tight')
#%% EFFETTI TERMINE LUNGHEZZA
plt.figure(figsize=(16, 16))
lengthplot_noise = 0.1

for i, l in enumerate([0.05, 0.5, 1, 3.2, 5.0, 7.0]):   
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", l, lengthplot_noise))
    create_case(x,
                x_guess,
                y,
                GPR.generate_kernel(GPR.kernel_gaussian,
                                    const = 1,
                                    length=l),
                R=lengthplot_noise)
    
plt.savefig('length_plot.png', bbox_inches='tight')


#%% PARAMETER OPTIMIZATION

gaus = GPR(x, y)


paramlist = {'length': np.linspace(0.1,100,100),
             'const': np.linspace(0.1, 10, 10)}

landscape, best_params = gaus.optimizer(param_dictionary=paramlist, noiselist = np.linspace(0.1,10,10), parallel = False)


#%%

#print("best parameters (probability, noise, length_scale):",lml.max(),r_opt, l_opt)
#
#create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_gaussian, length=l_opt), R=r_opt, title = "Parametri Otimizzati")
#plt.savefig('optimal_params.png', bbox_inches='tight')
#
