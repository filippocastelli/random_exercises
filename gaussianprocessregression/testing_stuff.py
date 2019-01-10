import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

import pickle

#%%
k2 = GPR.generate_kernel(kernel= GPR.kernel_periodic,length=3.126, period = 2*np.pi/0.7, const = 0.001)
k1 = GPR.generate_kernel(GPR.kernel_gaussian, length=3.127, const = 1)

k1(1,2)
def somma_wrap(*args, **kwargs):
    return k1(args[0], args[1]) +k2(args[0], args[1])

