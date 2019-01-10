import sympy as sp
sp.init_printing(use_latex = False, forecolor = 'White', fontsize = '100pt')
#%%

x1, x2, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 = sp.symbols('x_1, x_2, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11')

kernel1 = s1**2 *sp.exp(-(((x1 - x2)**2)/(2*(s2**2))))

k1der_1 = sp.diff(kernel1, s1)
k1der_2 = sp.diff(kernel1, s2)

kernel2 = s3**2*sp.exp(-(x1 - x2)**2 / (2*s4**2) - 2*sp.sin(sp.pi*(x1 - x2))**2 / s5**2)
k2der3 = sp.diff(kernel2, s3)
k2der4 = sp.diff(kernel2, s4)
k2der5 = sp.diff(kernel2, s5)

kernel3 = s6**2*(1+ (x1 -x2)**2 / (2* s8 * s7**2))**(-s8)

k3der6 = sp.diff(kernel3, s6)
k3der7 = sp.diff(kernel3, s7)
k3der8 = sp.diff(kernel3, s8)