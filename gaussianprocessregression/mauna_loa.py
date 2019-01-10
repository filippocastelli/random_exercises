import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt

import matplotlib.pylab as plt

from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Span, Label, Legend
from bokeh.io import output_notebook
from bokeh.palettes import brewer
output_notebook()

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

import pickle

#%%

data_monthly = pd.read_csv(pm.get_data("monthly_in_situ_co2_mlo.csv"), header=56)

# - replace -99.99 with NaN
data_monthly.replace(to_replace=-99.99, value=np.nan, inplace=True)

# fix column names
cols = ["year", "month", "--", "--", "CO2", "seasonaly_adjusted", "fit",
        "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]
data_monthly.columns = cols
cols.remove("--"); cols.remove("--")
data_monthly = data_monthly[cols]

# drop rows with nan
data_monthly.dropna(inplace=True)

# fix time index
data_monthly["day"] = 15
data_monthly.index = pd.to_datetime(data_monthly[["year", "month", "day"]])
cols.remove("year"); cols.remove("month")
data_monthly = data_monthly[cols]

data_monthly.head(5)

#%%

# function to convert datetimes to numbers that are useful to algorithms
#   this will be useful later when doing prediction

def dates_to_idx(timelist):
    reference_time = pd.to_datetime('1958-03-15')
    t = (timelist - reference_time) / pd.Timedelta(1, "Y")
    return np.asarray(t)

t = dates_to_idx(data_monthly.index)

# normalize CO2 levels
y = data_monthly["CO2"].values
first_co2 = y[0]
std_co2 = np.std(y)
y_n = (y - first_co2) / std_co2

data_monthly = data_monthly.assign(t = t)
data_monthly = data_monthly.assign(y_n = y_n)
#%%


# split into training and test set
sep_idx = data_monthly.index.searchsorted(pd.to_datetime("2003-12-15"))
data_early = data_monthly.iloc[:sep_idx+1, :]
data_later = data_monthly.iloc[sep_idx:, :]

#%%
plt.figure()
ax = plt.gca()
training_data_plot, = ax.plot(data_early['t'], data_early['y_n'], c = 'black')

test_data_plot, = ax.plot(data_later['t'], data_later['y_n'], c = 'red')

plt.title("Mauna Loa C02")
plt.legend([training_data_plot, test_data_plot], ["train", "test"])


#%%
k2 = GPR.generate_kernel(kernel= GPR.kernel_periodic,length=3.126, period = 2*np.pi/0.7, const = 0.001)
k1 = GPR.generate_kernel(GPR.kernel_gaussian, length=3.127)

k3 = GPR.generate_kernel(kernel=GPR.mix1, length=1, length2 = 2, period = 3, const = 4, const2 =1)

def somma_wrap(*args, **kwargs):
    return k1(args[0], args[1]) +k2(args[0], args[1])

def mixed_kernel(thetas):
    k1 = GPR.generate_kernel(kernel = GPR.kernel_gaussian, const =thetas[0], length = thetas[1])
    k2 = GPR.generate_kernel(kernel = GPR.kernel_periodic_decay

create_case(x,x_guess, y, kernel= k3, R = 0.1)


#%%
