import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr

#%%
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


 #%% REGRESSORE 1
k2 = GPR.generate_kernel(kernel= GPR.kernel_periodic,length=3.126, period = 2*np.pi/0.7, const = 0.001)
k1 = GPR.generate_kernel(GPR.kernel_gaussian, length=3.127)

k3 = GPR.generate_kernel(kernel=GPR.mix1, length=1, length2 = 2, period = 3, const = 4, const2 =1)

def somma_wrap(*args, **kwargs):
    return k1(args[0], args[1]) +k2(args[0], args[1])

create_case(x,x_guess, y, kernel= somma_wrap, R = 0.1)


#%%

