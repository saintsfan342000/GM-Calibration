import numpy as n
import matplotlib.pyplot as p
import figfun as f
import sympy as sp
from scipy.interpolate import griddata, interp1d

uni = 3
key = n.genfromtxt('../PTSummary.dat', delimiter=',')
projects = key[:,0].astype(int)
excludes = [10]
savedir = '../CalResults/Exclude_PT10'
projects = projects[ ~n.in1d(projects, excludes) ]

if len(excludes) > 0:
        printstr = '***!!!CHANGE YOUR SAVE DIRECTORY!!!***\n'
        print(printstr*5)
        ans = input('Have you changed the save-directory? yes or no:  '  )
        if ans not in ['yes', 'Yes', 'YES', 'y', 'Y']:
            raise ValueError('You need to makesure your savedirectory is properly set')

# Plastic work level
Wp = 1 #ksi

sr, sx, sq = sp.var('sr, sx, sq')
(cp12,cp13,cp21,cp23,cp31,cp32) = sp.var("cp12,cp13,cp21,cp23,cp31,cp32")
(cpp12,cpp13,cpp21,cpp23,cpp31,cpp32) = sp.var("cpp12,cpp13,cpp21,cpp23,cpp31,cpp32")
varlist = (cp12,cp13,cp21,cp23,cp31,cp32,cpp12,cpp13,cpp21,cpp23,cpp31,cpp32)

# Load up the yield function
# Line 0:  PHI
# Line 1: dPHI/dsq / dPHI/dsx
with open('Yld04.txt', 'r') as fid:
    PHI = eval(fid.readline())
    dPHI = eval(fid.readline())

# [0] SigX, sigQ, r
exp_sts = n.empty((1+len(projects),3))
# Uniaxial
# [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p, [5]eq_p, 
# [6]ez_p, [7]deqp/dexp, [8]deqp/dexp (moving)
d = n.genfromtxt('../Uniaxial/Uniax_6061_{}/CalData_Interp.dat'.format(uni), delimiter=',')
stsx, r = d[ d[:,0] == Wp, [1,7] ].ravel()
exp_sts[0] = stsx,0, r
SIGO = stsx
# PT
for k,x in enumerate(projects):
    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, 
    # [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru, [12]deqp/dexp (all Wp), [13]deqp/dexp (moving)
    d = n.genfromtxt('../GMPT-{}/CalData_Interp.dat'.format(x), delimiter=',')
    stsx, stsq, r = d[ d[:,1] == Wp, [2,3,12] ].ravel()
    exp_sts[k+1] = stsx,stsq, r

# Important note!  phi(0,0,SIGO) won't evaluate to 1, or to
# the locus's value at sq=0,sr=SIGO since the uniaxial yield sts SIGO
# doesn't land on the locus itself.  What that gives you is SIGO/intersection!
# In other words, all points on the locus evaluate to phi = 1.  But since the uniaxial point 
# SIGO isn't even on the locus.  

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
    
def makecont(res):
    '''
    Plot the contour
    '''
    # Do I determine SIGO now from the calibration at the sr=sq=0?
    x = n.linspace(0,1.2,100)
    y = x.copy()
    X,Y = n.meshgrid(x,y)
    sublist = [(varlist[i], res[i]) for i in range(12)]
    phi1 = PHI.subs(sublist)
    phi1 = (phi1/(4))**.125/SIGO
    phi = sp.lambdify((sr,sq,sx),phi1, 'numpy')
    Z = phi(0,X*SIGO,Y*SIGO)
    p.contour(Y,X,Z,levels=[1])

def plotexp():
    '''
    Plot experimental flow stresses
    '''
    l, = p.plot(exp_sts[:,0]/SIGO, exp_sts[:,1]/SIGO,'ro')
    return l

def plotnormals(pause=0.0):
    '''
    Plot r-values of experiments
    '''
    for s,q,r in exp_sts:
        dx = SIGO*0.04/n.sqrt(1+r**2)
        dy = r*dx
        if n.arctan2(dy,dx) >= n.pi/2:
            print('hey')
            dx*=-1
            dy*=-1
        x = n.array([s-dx,s, s+dx])/SIGO
        y = n.array([q-dy,q, q+dy])/SIGO
        p.plot(x,y,'-')
        p.pause(pause)
    return None


# I just want to get the surace locus for each for easy plotting
algs = ['CG', 'Nelder-Mead', 'BFGS', 'basinhopping']
for alg in algs:
    # [0]E value, [1]Uni-Sts Wt, [2]Uni-r wt, [3]rb wt, [4-15]cij...
    results = n.genfromtxt('{}/{}.dat'.format(savedir,alg), delimiter=',')
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
    
def plotpoint():
    loc = n.asarray(p.ginput(1)).ravel()
    #p.plot(loc[0],phi(0,loc[0]*SIGO,loc[1]*SIGO),'ro')
    print( phi(0,loc[0]*SIGO,loc[1]*SIGO))


