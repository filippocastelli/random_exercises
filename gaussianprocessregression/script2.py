import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr

 #%% REGRESSORE 1
k2 = GPR.generate_kernel(kernel= GPR.kernel_periodic,length=1.5, period = 2*np.pi/0.7)
k1 = GPR.generate_kernel(GPR.kernel_gaussian, length=3)
def somma_wrap(*args, **kwargs):
    return k1(args[0], args[1]) + 0.05*k2(args[0], args[1])
#%% EFFETTI TERMINE RUMORE

N = 10 # numero punti training
n = 500   # numero punti test
s = 0.1   # noise variance

period = 2*np.pi/0.7

rng = np.random.RandomState(5)
x = rng.uniform(-5, 5, size = (N,1))
x_guess = np.linspace(-5, 5, n)
y = f1(x) +s*np.random.randn(N)
plt.figure(figsize=(16, 16))
noiseplot_length = 3

#%%
for i, r in enumerate([0.00001, 0.02, 0.1, 0.8, 1.5, 5.0]):
    plt.subplot("32{}".format(i+1))
    plt.title("kernel={}, length={}, noise variance={}".format("gaussian", noiseplot_length, r))
    create_case(x,
                x_guess,
                y,
                kernel=somma_wrap,
                R=r)
#%%
plt.figure()
ax = plt.gca()
create_case(x = x,
            y= y,
            x_guess = x_guess,
            kernel = GPR.generate_kernel(GPR.kernel_periodic, length = 3, period = 2*np.pi/0.7),
            R = 0.001,
            title = "GPR",
            save = "GPR",
            orig_function = True)

cosine = ax.plot(x_guess, f1(x_guess), c="red")
#%% PARAMETER OPTIMIZATION

gaus = GPR(x = x,
           y = y,
           kernel = GPR.kernel_gaussian)

R_list = np.linspace(0.0, 1, 10)
L_list = np.linspace(0.1, 100, 10)
P_list = np.linspace(6, 10, 10)

lml, params= gaus.optimize2(R_list, L_list)
print("best parameters (log-likelihood, noise, length_scale):",lml.max(),params[0], params[1])
create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_gaussian, length=params[1]), R=params[0], title = "Parametri Otimizzati")


#%%

print("best parameters (probability, noise, length_scale, period):",lml.max(),params[0], params[1], params[2])

create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_periodic, length=params[1], period = params[2]), R=params[0], title = "Parametri Otimizzati")
plt.savefig('optimal_params.png', bbox_inches='tight')

#%% PARAMETER OPTIMIZATION

gaus2 = GPR(x = x,
           y = y,
           kernel = GPR.kernel_periodic)

R_list = np.linspace(0.0, 1, 100)
L_list = np.linspace(0.1, 100, 10)
P_list = np.linspace(6, 10, 10)

lml, params= gaus2.optimize2(R_list, L_list, P_list)

disp_params = np.round(params, 2)
print("best parameters (log-likelihood, noise, length_scale, period):",np.round(lml.max(),2),disp_params[0], disp_params[1], disp_params[2])
create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_periodic, length=params[1], period= params[2]), R=params[0], title = "Parametri Otimizzati")

