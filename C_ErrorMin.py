import numpy as n
from scipy.optimize import (minimize,
                            basinhopping, 
                            differential_evolution)
from scipy.interpolate import griddata
import sympy as sp
from sympy.utilities.autowrap import ufuncify
lam_or_ufun = 'ufuncify'
uni = 3
key = n.genfromtxt('../PTSummary.dat', delimiter=',')
projects = key[:,0].astype(int)

# Weights for flow sts, stn rat
w_s, w_e = 1, 0.1
# Plastic work level
Wp = 1 #ksi


sr, sx, sq = sp.var('sr, sx, sq')
(cp12,cp13,cp21,cp23,cp31,cp32) = sp.var("cp12,cp13,cp21,cp23,cp31,cp32")
(cpp12,cpp13,cpp21,cpp23,cpp31,cpp32) = sp.var("cpp12,cpp13,cpp21,cpp23,cpp31,cpp32")
varlist = (cp12,cp13,cp21,cp23,cp31,cp32,cpp12,cpp13,cpp21,cpp23,cpp31,cpp32)
cp44, cp55, cpp44, cpp55, cp66, cpp66 = 1, 1, 1, 1, 1, 1

# Line 0:  PHI
# Line 1: dPHI/dsq / dPHI/dsx
with open('Yld04.txt', 'r') as fid:
    PHI = eval(fid.readline())
    dPHI = eval(fid.readline())

# Uniaxial
# [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p, [5]eq_p, [6]ez_p, [7]deqp/dexp, [8]deqp/dexp (moving)
d = n.genfromtxt('../Uniaxial/Uniax_6061_{}/CalData_Interp.dat'.format(uni), delimiter=',')
stsx, r = d[ d[:,0] == Wp, [1,7] ].ravel()
SIGO = stsx
sublist = ((sx,stsx),(sq,0),(sr,0))
A = dPHI.subs(sublist)
B = (PHI.subs(sublist)/4)**(1/8)
E = w_e*(A/r - 1)**2 + w_s*(B/SIGO - 1)**2
print('Looping')
# PT
for k,x in enumerate(projects):
    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, 
    # [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru, [12]deqp/dexp (all Wp), [13]deqp/dexp (moving)
    d = n.genfromtxt('../GMPT-{}/CalData_Interp.dat'.format(x), delimiter=',')
    stsx, stsq, r = d[ d[:,1] == Wp, [2,3,12] ].ravel()
    sublist = ((sx,stsx),(sq,stsq),(sr,0))
    A = dPHI.subs(sublist)
    B = (PHI.subs(sublist)/4)**(1/8)
    E += w_e*(A/r - 1)**2 + w_s*(B/SIGO - 1)**2
print('differentiating')
# Derivatives of E w.r.t. each c
dE = [E.diff(i) for i in varlist]
print('lambdifying')

if lam_or_ufun == 'ufuncify':
    # ufuncify evaluations of fun(x) are ~40 microsec vs 720 microsec for lambdify
    # ufuncify evaluations of jac(x) are 1.1 microsec vs 25 microsec for lamdify
    # It's also slightly (5 to 10 %) to pass x[0], x[1],... instead of *x 
    F = ufuncify(varlist,E)
    # ufuncify can't return a list like lamdify can
    dF = [0]*12
    for i in range(12):
        dF[i] = ufuncify(varlist,dE[i])
    
    def fun(x):
        return F(x[0], x[1], x[2], x[3], x[4], x[5],
                x[6],x[7],x[8],x[9],x[10],x[11])
    
    def jac(x):
        return n.array([dF[i](*x) for i in range(12)])
else:
    F = sp.lambdify(varlist, E)
    dF = sp.lambdify(varlist, dE)
    
    def fun(x):
        return F(x[0], x[1], x[2], x[3], x[4], x[5],
                x[6],x[7],x[8],x[9],x[10],x[11])

    def jac(x):
        return n.array(dF(x[0], x[1], x[2], x[3], x[4], x[5],
                x[6],x[7],x[8],x[9],x[10],x[11]))

print('minimizing')
res1 = minimize(fun,
                  n.ones(12),
                  jac=jac,
                  method='Nelder-Mead',
                  options={'maxiter':10000, 'maxfev':10000})

res2 = minimize(fun,
                  n.ones(12),
                  jac=jac,
                  method='BFGS',
                  options={'maxiter':10000, 'maxfev':10000})

res3 = minimize(fun,
                  n.ones(12),
                  jac=jac,
                  method='CG',
                  options={'maxiter':10000, 'maxfev':10000})

res4 = basinhopping(fun,
                    n.ones(12),
                    niter=1000
                    )
[i.x for i in [res1,res2,res3,res4]]
[i.message for i in [res1,res2,res3,res4]]
[i.fun for i in [res1,res2,res3,res4]]

# Do I determine SIGO now from the calibration at the sr=sq=0?
x = n.linspace(0,1.2,100)*SIGO
y = x.copy()
X,Y = n.meshgrid(x,y)

sublist = [(varlist[i], res1.x[i]) for i in range(12)]

phi = PHI.subs(sublist)
phi*=.25
phi=phi**.125
phi*=(1/SIGO)
phi = sp.lambdify((sr,sq,sx),phi, 'numpy')

#Z = (phi(0,X,Y)/(4*SIGO))**(1/8)
Z = phi(0,X,Y)



import matplotlib.pyplot as p
p.contour(X,Y,Z,levels=1)



'''
import dill
dill.settings['recurse'] = True
fid = open('file', 'wb') # must have the b for binary!
dill.dump(fun, fid)
fid.close()
fid = open('file','rb')
fun = dill.load(fid)
fid.close()
'''
