# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:09:58 2019

@author: Kalle Kotka
"""

import numpy as np
from scipy.optimize import minimize

def rosen(x):
	"""Rosenbrock function"""
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
	""" Gradients for the Rosenbrock function """
	xm = x[1:-1]
	xm_m1 = x[:-2]
	xm_p1 = x[2:]
	der = np.zeros_like(x)
	der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
	der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
	der[-1] = 200*(x[-1]-x[-2]**2)
	return der

""" values for the x """
x0 = [ 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

""" Using Nelder-mead algorithm for minimizing the function"""
res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(res.x)
""" This nelder-mead method does not work for 10 dimensional problem and the exceeds the maximum number of evaluations """

""" Using Broyden-Fletcher-Goldfarb-Shanno algorithm for minimizing """
res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
print(res.x)
""" The Optimization terminated successfully for the BFGS algorithm """

