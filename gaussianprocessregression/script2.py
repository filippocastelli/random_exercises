import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr


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
#%% PARAMETER OPTIMIZATION - GAUSSIAN KERNEL
gaus = GPR(x = x,
           y = y,
           kernel = GPR.kernel_gaussian)

R_list = np.linspace(0.0, 1, 100)
L_list = np.linspace(0.1, 100, 100)
C_list = [1]

lml, params= gaus.optimizer(R_list, L_list,C_list)
print("best parameters (log-likelihood, noise, length_scale):",lml.max(),params[0], params[1])
create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_gaussian, length=params[1]), R=params[0], title = "Parametri Otimizzati - Kernel Gaussiano")

#%% PARAMETER OPTIMIZATION
gaus_periodic = GPR(x = x, y= y, kernel = GPR.kernel_periodic)


gaus2 = GPR(x = x,
           y = y,
           kernel = GPR.kernel_periodic)

#R_list = np.linspace(0.0, 0.05, 100)
R_list = np.linspace(0.01, 0.2, 100)
L_list = np.linspace(0.1, 15, 100)
#P_list = np.linspace(6, 8.9, 100)
P_list = [8.78]
C_list = [1]

lml, params= gaus2.optimizer(R_list, L_list, P_list,C_list)

disp_params = np.round(params, 2)
print("best parameters (log-likelihood, noise, length_scale, period, const):",np.round(lml.max(),2),disp_params[0], disp_params[1], disp_params[2], disp_params[3])
create_case(x, x_guess, y, kernel= GPR.generate_kernel(GPR.kernel_periodic, length=params[1], period= params[2]), R=params[0], title = "Parametri Otimizzati - Kernel Periodico")

#%%
gausmix = GPR(x = x,
           y = y,
           kernel = GPR.mix1)

#R_list = np.linspace(0.0, 0.5, 10)
#L_list = np.linspace(0.0, 10, 100)
#C1_list = np.linspace(0.1, 10, 5)
#L2_list = np.linspace(0.1, 10, 100)
#P_list = np.linspace(0.1, 15, 100)
#C2_list = np.linspace(0.1, 10, 5)

R_list = [0.01]
L_list = np.linspace(3, 10, 100)
C1_list = np.linspace(0,1,10)
L2_list = [3]
P_list = [8.78]
C2_list = [0.]


lml, params= gausmix.optimizer(R_list, L_list, C1_list, L2_list, P_list, C2_list)

b_noise, b_l, b_c1, b_l2, b_p, b_c2 = params

disp_params = np.round(params, 2)
print("best parameters (log-likelihood, noise, (length_scale, const), (length_scale, period, const)):",np.round(lml.max(),2), disp_params)
create_case(x,
            x_guess,
            y,
            kernel= GPR.generate_kernel(GPR.mix1,
                                        length=b_l,
                                        const = b_c1,
                                        length2 = b_l2,
                                        period=b_p,
                                        const2 = b_c2),
            R=b_noise,
            title = "Parametri Otimizzati")
            
 #%% REGRESSORE 1
k2 = GPR.generate_kernel(kernel= GPR.kernel_periodic,length=3.126, period = 2*np.pi/0.7, const = 0.001)
k1 = GPR.generate_kernel(GPR.kernel_gaussian, length=3.127)

k3 = GPR.generate_kernel(kernel=GPR.mix1, length=1, length2 = 2, period = 3, const = 4, const2 =1)

def somma_wrap(*args, **kwargs):
    return k1(args[0], args[1]) +k2(args[0], args[1])

create_case(x,x_guess, y, kernel= k3, R = 0.1)