import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr
#%% IMPOSTAZIONE PROBLEMA

#f = lambda x: np.cos(.7*x).flatten()

N = 5     # numero punti training
n = 500   # numero punti test
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
#%%
# PLOT PRIORI

prior(x = x,
      x_guess = x_guess,
      kernel = GPR.kernel_gaussian)
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
            title = "GPR",
            save = "GPR",
            orig_function = True)

cosine = ax.plot(x_guess, f1(x_guess), c="red")

#%% EFFETTI TERMINE RUMORE
plt.figure(figsize=(16, 16))
noiseplot_length = 3

for i, r in enumerate([0.0001, 0.03, 0.09, 0.8, 1.5, 5.0]):
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", noiseplot_length, r))
    create_case(x,
                x_guess,
                y,
                GPR.generate_kernel(GPR.kernel_gaussian, length=noiseplot_length),
                R=r)
    
plt.savefig('noise_plot.png',bbox_inches='tight')
#%% EFFETTI TERMINE LUNGHEZZA
plt.figure(figsize=(16, 16))
lengthplot_noise = 1e-3

for i, l in enumerate([0.05, 0.5, 1, 3.2, 5.0, 7.0]):   
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", l, lengthplot_noise))
    create_case(x,
                x_guess,
                y,
                GPR.generate_kernel(GPR.kernel_gaussian, length=l),
                R=lengthplot_noise)
    
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
create_case(x,
            x_guess,
            y,
            GPR.generate_kernel(GPR.kernel_gaussian, length=best_params[2]),
            R=best_params[1],
            title = "Parametri Otimizzati")
plt.savefig('optimal_params.png', bbox_inches='tight')
#%%

gaus = GPR(x, y)
R_list = np.linspace(0.0, 10, 100)
L_list = np.linspace(0.1, 100, 100)
lml, r_opt, l_opt= gaus.optimize2(R_list, L_list)

print("best parameters (probability, noise, length_scale):",lml.max(),r_opt, l_opt)
#plt.figure()
#plt.title("Prob. history")
#plt.plot(history[:,1], history[:,2])
#print("best parameters (probability, r, b): ", best_params)
#plt.show()

plt.figure()
plt.imshow(lml, cmap = 'viridis')
plt.figure()
create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_gaussian, length=l_opt), R=r_opt, title = "Parametri Otimizzati")
plt.savefig('optimal_params.png', bbox_inches='tight')