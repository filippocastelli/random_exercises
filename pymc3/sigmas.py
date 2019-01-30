import theano.tensor as tt
import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#%%
SEED = 326402

np.random.seed(SEED)


#%%

dati = pd.DataFrame(index = ['gene0','gene1', 'gene2', 'gene3','gene4'],
                    data = {'paziente1':    [0,1,0,0,0],
                            'paziente2':    [0,1,0,0,0],
                            'paziente3':    [1,0,1,1,0],                                        
                            'paziente4':    [1,0,1,1,0],
                            'paziente5':    [0,1,0,0,0],
                            'paziente6':    [0,1,0,0,0],
                            'paziente7':    [0,0,0,0,0],
                            'paziente8':    [0,1,1,1,0],
                            'paziente9':    [0,1,0,0,0],
                            'paziente10':   [0,0,1,1,0],
                            
                            }
                    )

#maybe bad label format?
#dati.replace(0, -1, inplace = True)
#%%

dati_t = dati.transpose()
#plt.figure()

plt.matshow(dati_t.cov())
plt.title('correlation plot')

#plt.figure()
plt.matshow(dati_t.corr())
plt.title('covariance plot')

#%%
    
model = pm.Model()

    
SEED = 12345

with model:
    packed_L = pm.LKJCholeskyCov('Packed_L', n=len(dati), eta = 1.5., sd_dist = pm.HalfCauchy.dist(1))
    L = pm.expand_packed_triangular(len(dati), packed_L)
#    sigma = pm.Wishart("Sigma", nu = 1, V = sigma0, shape = (3,3))
    sigma = pm.Deterministic('Sigma', L.dot(L.T))
#    mu = pm.Normal('mu', 0., 10., shape=3, testval = dati.mean(axis=1))
    logits = pm.MvNormal("Logits", mu = 0*np.ones(len(dati)), cov = sigma, shape = len(dati))
    p = pm.Deterministic('p', tt.exp(logits)/(1 + tt.exp(logits)))
    observed = pm.Binomial("Observed", p = p, n = 1, observed = dati.values.T)
    trace  = pm.sample(random_seed = SEED, cores = 1)
    
    
    
#%%

sigmas = trace['Sigma']

plt.matshow(sigmas.mean(axis = 0))
plt.matshow(sigmas.std(axis = 0))