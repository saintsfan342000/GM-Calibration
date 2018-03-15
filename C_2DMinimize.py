import numpy as n
from scipy.optimize import (minimize,
                            basinhopping, 
                            differential_evolution)
from scipy.interpolate import griddata, interp1d
import sympy as sp
from sympy import Rational, Matrix
import matplotlib.pyplot as p
from sympy.utilities.autowrap import ufuncify, autowrap

'''
Calibrates the YLD02-2D yield function
'''

# User parameters
lam_or_auto = 'auto'
# Specify any expts you want to exclude
excludes = [10]
# Specify whether you want to special weght PT equibiax and uniax
weight_ptuniax = True
weight_equibiax = True
# Where the results are being saved
savedir = '../CalResults/2D/AllNormal'

uni = 3 # Uniax expt no
equibiax = 8 # PT Equibix no
PTuniax = 11 # PT uniax (pure hoop)

key = n.genfromtxt('../PTSummary.dat', delimiter=',')
projects = key[:,0].astype(int)
projects = projects[ ~n.in1d(projects, excludes) ]
if len(excludes) > 0:
        printstr = '***!!!CHANGE YOUR SAVE DIRECTORY!!!***\n'
        print(printstr*5)
        ans = input('Have you changed the save-directory? \n' +
                    'Currently set to: {} yes or no:  '.format(savedir)  )
        if ans not in ['yes', 'Yes', 'YES', 'y', 'Y']:
            raise ValueError('You need to makesure your savedirectory is properly set')
        
# Weights for flow sts, stn rat
w_s, w_e = 1, 0.1
# Weight amplification factors for uniaixial sts, stn, and balanced biaxial
wu_s, wu_e, wrb = sp.symbols('wu_s, wu_e, wrb', positive = True)
# Plastic work level
Wp = 1 #ksi

# Symbols to generate the Yield Function 
sr, sx, sq = sp.var('sr, sx, sq')
a1,a2,a3,a4,a5,a6 = sp.var('a1,a2,a3,a4,a5,a6')
a7,a8 = 1,1
varlist = (a1,a2,a3,a4,a5,a6)

Tp = Matrix(5,3,[2,0,0,-1,0,0,0,-1,0,0,2,0,0,0,3])*Rational(1,3)
Tpp = Matrix(5,5,[-2,2,8,-2,0,1,-4,-4,4,0,4,-4,-4,1,0,-2,8,2,-2,0,0,0,0,0,9])
Tpp*=Rational(1,9)

Lp = Tp*Matrix([a1,a2,a7])
Lp = Matrix(3,3,[Lp[0],Lp[1],0,Lp[2],Lp[3],0,0,0,Lp[4]])

Lpp = Tpp*Matrix([a3,a4,a5,a6,a8])
Lpp = Matrix(3,3,[Lpp[0],Lpp[1],0,Lpp[2],Lpp[3],0,0,0,Lpp[4]])

S = Matrix([sq, sx, 0])

Xp = (Lp*S)[:2] #I get away with this because I only have principle stresses!
Xpp = (Lpp*S)[:2]

a = 8
PHI = ( (Xp[0]-Xp[1])**a + 
        (2*Xpp[0]+Xpp[1])**a + 
        (2*Xpp[1]+Xpp[0])**a
      )

dPHI = PHI.diff(sq)/PHI.diff(sx)

# Now Load up experimental data and build up the error function
# [0] SigX, sigQ, r
exp_sts = n.empty((1+len(projects),3))
# Uniaxial
# [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p, [5]eq_p, 
# [6]ez_p, [7]deqp/dexp, [8]deqp/dexp (moving)
d = n.genfromtxt('../Uniaxial/Uniax_6061_{}/CalData_Interp.dat'.format(uni), delimiter=',')
stsx, r = d[ d[:,0] == Wp, [1,7] ].ravel()
exp_sts[0] = stsx,0, r
SIGO = stsx
sublist = ((sx,stsx),(sq,0),(sr,0)) # Having sr doesn't do anything
A = dPHI.subs(sublist)
B = (PHI.subs(sublist)/4)**(1/8)
E = wu_e*w_e*(A/r - 1)**2 + wu_s*w_s*(B/SIGO - 1)**2
E_raw = (A/r - 1)**2 + (B/SIGO - 1)**2
print('Looping')
# PT
for k,x in enumerate(projects):
    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, 
    # [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru, [12]deqp/dexp (all Wp), [13]deqp/dexp (moving)
    d = n.genfromtxt('../GMPT-{}/CalData_Interp.dat'.format(x), delimiter=',')
    stsx, stsq, r = d[ d[:,1] == Wp, [2,3,12] ].ravel()
    exp_sts[k+1] = stsx,stsq, r
    sublist = ((sx,stsx),(sq,stsq),(sr,0))
    A = dPHI.subs(sublist)
    B = (PHI.subs(sublist)/4)**(1/8)
    # Exception for balanced biaxial:  Give it wrb
    if (x == equibiax) and (weight_equibiax):
        E += wrb*w_e*(A/r - 1)**2 + w_s*(B/SIGO - 1)**2
    # Exception for pure hoop sts:  give it Wu_e and wu_s
    elif (x == PTuniax) and (weight_ptuniax):
        E += wu_e*w_e*(A/r - 1)**2 + wu_s*w_s*(B/SIGO - 1)**2
    else:
        E += w_e*(A/r - 1)**2 + w_s*(B/SIGO - 1)**2
    E_raw += (A/r - 1)**2 + (B/SIGO - 1)**2

print('Differentiating')
# Derivatives of E w.r.t. each c
dE = [E.diff(i) for i in varlist]

if lam_or_auto == 'auto':
    print('Autowrapping!')
    # Evlau times for fun(x):
    # lamdify:  ~700 micsec.  ufuncify:  ~40 micsec.  autowrap:  ~7 micsec
    # Eval times for jac(x)
    # lambdify:  ~25 *milli*sec.  ufuncify:  1 *mili*sec autowrap:  ~120 micsec
    # ufuncify evaluations of jac(x) are 1.1 microsec vs 25 millisec for lamdify
    # It's also slightly (5 to 10 %) to pass x[0], x[1],... instead of *x 
    F = autowrap(E, args=(*varlist, wu_s, wu_e, wrb))
    E_raw_val = autowrap(E_raw, args=varlist)
    # autowrap can't return a list like lamdify can
    dF = [0]*len(varlist)
    for i in range(len(varlist)):
        dF[i] = autowrap(dE[i], args=(*varlist, wu_s, wu_e, wrb))
    
    def fun(x, wu_s, wu_e, wrb):
        return F(x[0], x[1], x[2], x[3], x[4], x[5], wu_s, wu_e, wrb)
    
    def jac(x, wu_s, wu_e, wrb):
        return n.array([dF[i](*x, wu_s, wu_e, wrb) for i in range(len(varlist))])

else:
    print('Lambdifying :(')
    F = sp.lambdify((*varlist, wu_s, wu_e, wrb), E)
    dF = sp.lambdify((*varlist, wu_s, wu_e, wrb), dE)
    E_raw_val = sp.lambdify(*varlist, E_raw)
    def fun(x, wu_s, wu_e, wrb):
        return F(x[0], x[1], x[2], x[3], x[4], x[5],wu_s, wu_e, wrb)

    def jac(x, wu_s, wu_e, wrb):
        return n.array(dF(x[0], x[1], x[2], x[3], x[4], x[5],wu_s, wu_e, wrb))

    
# Specify algorithm, weights
algs = ['Nelder-Mead', 'BFGS', 'CG']# 'basinhopping']

# Generic minimizer call
def callmin(alg, wu_s, wu_e, wrb):
    if alg in ['Nelder-Mead', 'BFGS', 'CG']:
        res = minimize(fun,
                      x0 = n.ones(len(varlist)),
                      args = (wu_s, wu_e, wrb),
                      jac=jac,
                      method=alg,
                      options={'maxiter':10000, 'maxfev':10000})
    else:
        res = basinhopping(fun,
                    x0 = n.ones(len(varlist)),
                    minimizer_kwargs={'args':(wu_s, wu_e, wrb)},
                    niter=100
                    )
    return res

def getcontour(res,N=100):
    '''
    Just return the locus as x,y coords.  Don't plot
    '''
    x = n.linspace(0,1.2,N)
    y = x.copy()
    X,Y = n.meshgrid(x,y)
    sublist = [(varlist[i], res[i]) for i in range(len(varlist))]
    phi1 = PHI.subs(sublist)
    phi1 = (phi1/(4))**.125/SIGO
    phi = sp.lambdify((sr,sq,sx),phi1, 'numpy')
    Z = phi(0,X*SIGO,Y*SIGO)
    p.figure()
    c = p.contour(Y,X,Z,levels=[1])
    p.close()
    return c.allsegs[0][0]

# Result files
# Loop through algorithms and weights
# Saving AlgName.dat with score, weights, and coefficients
# Then saving AlgName.npy with locus for plotting
for alg in algs:
    print('Minimizing {}'.format(alg))
    results = n.empty((27,4+len(varlist)))
    row = 0
    for wus in [100,10,1]:
        for wue in [100,10,1]:
            for wrb in [100,10,1]:
                res = callmin(alg, wus, wue, wrb)
                results[row] = [E_raw_val(*(res.x)), wus, wue, wrb, *(res.x)]
                row += 1
    n.savetxt('{}/{}.dat'.format(savedir, alg),
            X = results,
            header = '[0]E value, [1]Uni-Sts Wt, [2]Uni-r wt, [3]rb wt, [4-15]cij...',
            fmt = '%.6f, %.0f, %.0f, %.0f, ' + '%.6f, '*(len(varlist)-1) + '%.6f'
            )
    # I just want to get the surace locus for each for easy plotting
    N = 200
    # A 3D array of the locus for each of the results
    locus = n.empty((results.shape[0],N,2))
    for z in range(results.shape[0]):
        # Annoyingly, the number of points in each locus changes
        tempxy = getcontour(results[z,4:], N)
        # So I need in interpolate based on the angle (since it is single-valued)
        q = n.arctan2(tempxy[:,1],tempxy[:,0])
        qspace = n.linspace(q.min(),q.max(),N)
        locus[z,:,:] = interp1d(q,tempxy, axis=0, fill_value='extrapolate').__call__(qspace)
    n.save('{}/{}-locus.npy'.format(savedir,alg), locus)






'''
Dilling works for lambdified expressions, but not ufuncified expressions

import dill
dill.settings['recurse'] = True
fid = open('file', 'wb') # must have the b for binary!
dill.dump(fun, fid)
fid.close()
fid = open('file','rb')
fun = dill.load(fid)
fid.close()

But for ufuncified expressions, you can pass the kwarg tempdir
with which you specify a directory to keep the files it generates 
One of these will be called "wrapper_module_n", where n is 0 or 1.
In a different session, you can "import wrapper_module_n".
The ufuncified expression is called as wrapper_module_n.autofunc(arg1,argw...)

see https://stackoverflow.com/questions/1260792/import-a-file-from-a-subdirectory
for how I can stick those into a subdir to keep main dir tidy

'''
