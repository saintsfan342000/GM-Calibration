import numpy as n
from scipy.optimize import (minimize,
                            basinhopping, 
                            differential_evolution)
from scipy.interpolate import griddata, interp1d
import sympy as sp
import matplotlib.pyplot as p
from sympy.utilities.autowrap import ufuncify, autowrap

'''
Calibrates the YLD04-3D yield function
'''

# User parameters
lam_or_auto = 'auto'
uni = 3
eqbiax = 8
# Specify any expts you want to exclude
excludes = [10]
# Where the results are being saved
savedir = '../CalResults/Weight_BothUni2'

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
(cp12,cp13,cp21,cp23,cp31,cp32) = sp.var("cp12,cp13,cp21,cp23,cp31,cp32")
(cpp12,cpp13,cpp21,cpp23,cpp31,cpp32) = sp.var("cpp12,cpp13,cpp21,cpp23,cpp31,cpp32")
varlist = (cp12,cp13,cp21,cp23,cp31,cp32,cpp12,cpp13,cpp21,cpp23,cpp31,cpp32)
cp44, cp55, cpp44, cpp55, cp66, cpp66 = 1, 1, 1, 1, 1, 1

Cp = sp.zeros(6,6)
Cp[0,1], Cp[0,2] = -cp12, -cp13
Cp[1,0], Cp[1,2] = -cp21, -cp23
Cp[2,0], Cp[2,1] = -cp31, -cp32
Cp[3,3], Cp[4,4], Cp[5,5] = cp44, cp55, cp66

Cpp = sp.zeros(6,6)
Cpp[0,1], Cpp[0,2] = -cpp12, -cpp13
Cpp[1,0], Cpp[1,2] = -cpp21, -cpp23
Cpp[2,0], Cpp[2,1] = -cpp31, -cpp32
Cpp[3,3], Cpp[4,4], Cpp[5,5] = cpp44, cpp55, cpp66

T = sp.zeros(6,6)
T[0,0], T[0,1], T[0,2] = 2, -1, -1
T[1,0], T[1,1], T[1,2] = -1, 2, -1
T[2,0], T[2,1], T[2,2] = -1, -1, 2
T[3,3], T[4,4], T[5,5] = 3, 3, 3
T*=sp.Rational(1,3)

s = sp.Matrix([sr, sq, sx, 0, 0, 0])
Sp = (Cp*T*s)[:3]
Spp = (Cpp*T*s)[:3]

a = 8
PHI = ( (Sp[0]-Spp[0])**a + 
      (Sp[0]-Spp[1])**a + 
      (Sp[0]-Spp[2])**a + 
      (Sp[1]-Spp[0])**a + 
      (Sp[1]-Spp[1])**a + 
      (Sp[1]-Spp[2])**a + 
      (Sp[2]-Spp[0])**a + 
      (Sp[2]-Spp[1])**a + 
      (Sp[2]-Spp[2])**a
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
sublist = ((sx,stsx),(sq,0),(sr,0))
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
    if x == 8:
        E += wrb*w_e*(A/r - 1)**2 + w_s*(B/SIGO - 1)**2
    # Exception for pure hoop sts:  give it Wu_e and wu_s
    if x == 11:
        E += wu_e*w_e*(A/r - 1)**2 + wu_s*w_s*(B/SIGO - 1)**2
    else:
        E += w_e*(A/r - 1)**2 + w_s*(B/SIGO - 1)**2
    E_raw += (A/r - 1)**2 + (B/SIGO - 1)**2

print('differentiating')
# Derivatives of E w.r.t. each c
dE = [E.diff(i) for i in varlist]

if lam_or_auto == 'auto':
    print('autowrapping!')
    # Evlau times for fun(x):
    # lamdify:  ~700 micsec.  ufuncify:  ~40 micsec.  autowrap:  ~7 micsec
    # Eval times for jac(x)
    # lambdify:  ~25 *milli*sec.  ufuncify:  1 *mili*sec autowrap:  ~120 micsec
    # ufuncify evaluations of jac(x) are 1.1 microsec vs 25 millisec for lamdify
    # It's also slightly (5 to 10 %) to pass x[0], x[1],... instead of *x 
    F = autowrap(E, args=(*varlist, wu_s, wu_e, wrb))
    E_raw_val = autowrap(E_raw, args=varlist)
    # autowrap can't return a list like lamdify can
    dF = [0]*12
    for i in range(12):
        dF[i] = autowrap(dE[i], args=(*varlist, wu_s, wu_e, wrb))
    
    def fun(x, wu_s, wu_e, wrb):
        return F(x[0], x[1], x[2], x[3], x[4], x[5],
                x[6],x[7],x[8],x[9],x[10],x[11], wu_s, wu_e, wrb)
    
    def jac(x, wu_s, wu_e, wrb):
        return n.array([dF[i](*x, wu_s, wu_e, wrb) for i in range(12)])

else:
    print('lambdifying :(')
    F = sp.lambdify((*varlist, wu_s, wu_e, wrb), E)
    dF = sp.lambdify((*varlist, wu_s, wu_e, wrb), dE)
    E_raw_val = sp.lambdify(*varlist, E_raw)
    def fun(x, wu_s, wu_e, wrb):
        return F(x[0], x[1], x[2], x[3], x[4], x[5],
                x[6],x[7],x[8],x[9],x[10],x[11], 
                wu_s, wu_e, wrb)

    def jac(x, wu_s, wu_e, wrb):
        return n.array(dF(x[0], x[1], x[2], x[3], x[4], x[5],
                x[6],x[7],x[8],x[9],x[10],x[11], 
                wu_s, wu_e, wrb))

    
# Specify algorithm, weights
algs = ['Nelder-Mead', 'BFGS', 'CG', 'basinhopping']

# Generic minimizer call
def callmin(alg, wu_s, wu_e, wrb):
    if alg in ['Nelder-Mead', 'BFGS', 'CG']:
        res = minimize(fun,
                      x0 = n.ones(12),
                      args = (wu_s, wu_e, wrb),
                      jac=jac,
                      method=alg,
                      options={'maxiter':10000, 'maxfev':10000})
    else:
        res = basinhopping(fun,
                    x0 = n.ones(12),
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
    sublist = [(varlist[i], res[i]) for i in range(12)]
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
    results = n.empty((27,16))
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
            fmt = '%.6f, %.0f, %.0f, %.0f, ' + '%.6f, '*11 + '%.6f'
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
