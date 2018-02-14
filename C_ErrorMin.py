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

# Exp. stresses
exp_sts = n.empty((1+len(projects),2))

# Uniaxial
# [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p, [5]eq_p, [6]ez_p, [7]deqp/dexp, [8]deqp/dexp (moving)
d = n.genfromtxt('../Uniaxial/Uniax_6061_{}/CalData_Interp.dat'.format(uni), delimiter=',')
stsx, r = d[ d[:,0] == Wp, [1,7] ].ravel()
exp_sts[0] = stsx,0
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
    exp_sts[k+1] = stsx,stsq
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
#  Much slower if fun is lamdified b/c this makes over 100,000 function evals!
res4 = basinhopping(fun,
                    n.ones(12),
                    niter=100
                    )

[i.x for i in [res1,res2,res3,res4]]
[i.message for i in [res1,res2,res3,res4]]
[i.fun for i in [res1,res2,res3,res4]]
[i.nfev for i in [res1,res2,res3,res4]]

# Do I determine SIGO now from the calibration at the sr=sq=0?
x = n.linspace(0,1.2,100)
y = x.copy()
X,Y = n.meshgrid(x,y)

sublist = [(varlist[i], res4.x[i]) for i in range(12)]

phi1 = PHI.subs(sublist)
phi1 = (phi1/(4))**.125/SIGO
phi = sp.lambdify((sr,sq,sx),phi1, 'numpy')

Z = phi(0,X*SIGO,Y*SIGO)
# Important note!  phi(0,0,SIGO) won't evaluate to 1, or to
# the locus's value at sq=0,sr=SIGO since the uniaxial yield sts SIGO
# doesn't land on the locus itself.  What that gives you is SIGO/intersection!
# In other words, all points on the locus evaluate to phi = 1.  But since the uniaxial point 
# SIGO isn't even on the locus.  

import matplotlib.pyplot as p

p.contour(X,Y,Z,levels=[1])
p.plot(exp_sts[:,1]/SIGO, exp_sts[:,0]/SIGO,'ro')

def plotpoint():
    loc = n.asarray(p.ginput(1)).ravel()
    #p.plot(loc[0],phi(0,loc[0]*SIGO,loc[1]*SIGO),'ro')
    print( phi(0,loc[0]*SIGO,loc[1]*SIGO))
'''
[array([0.961635, 1.1259, 1.06189, 0.529042, 1.10732, 0.503277, 0.119869, 0.559774, 1.21067, 1.21762, 1.08453, 1.284]),
 array([-0.871706, -0.588812, 0.871706, 0.282894, -0.234004, -0.194853, -0.131771, 1.45379, -0.327191, -1.36011, -0.449007, 0.665625]),
 array([0.00552604, 0.989272, 0.995557, -0.0058843, -0.000759139, 0.984147, 0.00552604, 0.989272, 0.995557, -0.0058843, -0.000759139, 0.984147]),
 array([2.43932, -0.0839988, -2.44011, -2.52375, -1.10794, 0.148537, 2.58464, 0.673638, -1.97157, -0.961163, 1.12249, 1.57473])]
'''


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
