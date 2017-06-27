import numpy as n
from numpy import (dstack, vstack, hstack,
                   linspace, array, nanmean, nanstd)
from numpy.linalg import eigvalsh, eigh
import matplotlib.pyplot as p
from scipy.interpolate import griddata, interp1d
from scipy.io import loadmat
from pandas import read_csv
from mysqrtm import mysqrtm
from CircleFitByPratt import CircleFitByPratt as CF
import os

'''
Investigate the difference between calculating log strains by
(1) n.log(U)
(2) n.log(eigU) then rotating back
(3) n.log(F22-1)
(4) n.log(Virst extensometer over /- 0.1")
'''
projects = [1,2,3,4,7,8,10,11]
prefix = 'GMPT'

if prefix == 'TTGM':
    ht = .05 # the +/- height that defines my gagelen
    thickness = 0.04
    key = n.genfromtxt('../../TTSummary.dat', delimiter=',', usecols=(0,3))
elif prefix == 'GMPT':
    ht = .5
    thickness = 0.05
    key = n.genfromtxt('../../PTSummary.dat', delimiter=',', usecols=(0,4))
else:
    raise ValueError('Invalid prefix name given')


os.chdir('..') # Back out to AA_PyScripts

for expt in projects:
    proj = '{}-{}'.format(prefix,expt)

    print(proj)
    alpha = key[ key[:,0]==expt ].ravel()[1]


    expt = int( proj.split('_')[0].split('-')[1])

    os.chdir('../{}'.format(proj))
    #Initialize a few things before looping and calculating every stage

    #Just up to the limitload
    PS = n.genfromtxt('prof_stages.dat', delimiter=',', dtype=int)[:3]
    upperlim = .5*(PS[-1]+PS[-2])
    PS = n.linspace(PS[0],upperlim, 10, dtype=int)


    datalist = ['NEx_aram', 'NEy_aram', 'NExy_aram', 'G_aram',
                   'NEx_alt', 'NEy_alt', 'G_alt',
                   'ex_U', 'ey_U',
                   'ex_F', 'ey_F', 
                   'ex_eigU', 'ey_eigU']
    data_mean = n.empty((len(PS), len(datalist)+5), dtype=n.float64)
    data_med = n.empty_like(data_mean)

    #Cycle through the stages
    for k,stage in enumerate(PS):
        data_mean[k,0] = stage
        data_med[k,0] = stage
        # Load up the stage
        name = 'stage_{}'.format(stage)
        A = loadmat('{}'.format(proj), variable_names=name)[name]
        #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)

        Q = n.arctan2(A[:,2], A[:,4])*180/n.pi
        q_rng = Q.max()-Q.min()
        q_mid = Q.min()+q_rng/2
          
        rng = (n.abs(A[:,3])<=ht) & (Q>=q_mid-q_rng/4) & (Q<=q_mid+q_rng/4)
        
        F=A[rng,-4:].reshape(-1,2,2)   # A "stack" of 2x2 deformation gradients 
        FtF = n.einsum('...ji,...jk',F,F)
        U = mysqrtm( FtF )     #Explicit calculation of sqrtm

        # Filter
        if proj[:4] == 'TTGM':
            if not n.isnan(alpha):
                rat = (F[:,1,1]-1) / n.arctan(F[:,0,1]/F[:,1,1]);
            else:
                rat = (F[:,0,0]-1) / (F[:,1,1]-1)
        else:
            rat = (F[:,0,0]-1) / (F[:,1,1]-1)
        
        ratmean = rat.mean()
        ratdev = rat.std()
        keeps = (rat>=ratmean-1.5*ratdev) & (rat<=ratmean+1.5*ratdev)

        F = F.compress(keeps, axis=0)
        U = U.compress(keeps, axis=0)

        #Filter again
        if proj[:4] == 'TTGM':
            if not n.isnan(alpha):
                rat = (F[:,1,1]-1) / n.arctan(F[:,0,1]/F[:,1,1]);
            else:
                rat = (F[:,0,0]-1) / (F[:,1,1]-1)
        else:
            rat = (F[:,0,0]-1) / (F[:,1,1]-1)
        
        ratmean = rat.mean()
        ratdev = rat.std()
        keeps = (rat>=ratmean-0.5*ratdev) & (rat<=ratmean+0.5*ratdev)

        F = F.compress(keeps, axis=0)
        U = U.compress(keeps, axis=0)

        NEx_aram = U[:,0,0] - 1
        NEy_aram = U[:,1,1] - 1
        NExy_aram = U[:,0,1]
        G_aram = n.arctan(NExy_aram/(1+NEx_aram)) + n.arctan(NExy_aram/(1+NEy_aram))
        NEx_alt = F[:,0,0]-1
        NEy_alt = F[:,1,1]-1
        G_alt = n.arctan(F[:,0,1]/F[:,1,1]);
        
        ex_U, ey_U = n.log(U[:,0,0]), n.log(U[:,1,1])
        ex_F, ey_F = n.log(F[:,0,0]), n.log(F[:,1,1])
       
        eigU, V = eigh(U)
        LEprin = n.log(eigU)
        # Now rotate the principle log strains back to x, y using the eigenvectors
        LErot = n.einsum('...ij,...j,...jk',V, LEprin, V)
        ex_eigU, ey_eigU = LErot[:,0,0], LErot[:,1,1] 
       
        # Interp at +/- ht" undeformed for axial
        rng = (Q>=q_mid-q_rng/4) & (Q<=q_mid+q_rng/4)
        rng =  (
                rng &
                (
                    ((A[:,3]>=ht-thickness) & (A[:,3]<=ht+thickness)) |
                    ((A[:,3]<=-ht+thickness) & (A[:,3]>=-ht-thickness))
                )
               )
        xspace = linspace( A[rng,2].min()+thickness/2, A[rng,2].max()-thickness/2, 
                  2*len(n.unique(A[rng,0])) )
        x_hi, x_lo = griddata( A[rng,2:4], A[rng,6],
                        (xspace[None,:],array([[ht],[-ht]])),
                        method='linear').mean(axis=1)
        # Assign
        ext_eps = (x_hi-x_lo)/(2*ht)-1
        ext_e = n.log(ext_eps+1)

        ## BFkC
        # Five circles btwn ht
        rng = (Q>=q_mid-45) & (Q<=q_mid+45)
        yspace = linspace(ht,-ht,5)
        R, Ro = 0,0
        for z,y in enumerate(yspace):
            rng2 = rng & (A[:,3]>=y-thickness) & (A[:,3]<=y+thickness)
            xspace = linspace( A[rng2,2].min()+thickness/2, A[rng2,2].max()-thickness/2,
                    2*len(n.unique(A[rng2,0])))
            xo,yo,zo,X,Y,Z = griddata(A[rng2,2:4], A[rng2,2:8],
                            (xspace[None,:], n.array([[y]])),
                            method='linear')[0].T
            Ro += CF(n.c_[xo,zo])[-1]
            R += CF(n.c_[X,Z])[-1]

        cf_epsq = R/Ro-1
        cf_eq = n.log(R/Ro)

        for z,d in enumerate(datalist):
            data_mean[k,z+1] = eval('{}.mean()'.format(d))
            data_med[k,z+1] = eval('n.median({})'.format(d))

        data_mean[k,-4:] = ext_eps, ext_e, cf_epsq, cf_eq
        data_med[k,-4:] = ext_eps, ext_e, cf_epsq, cf_eq

    if n.any(n.isnan(data_mean)):
        print('Getting nans')

    if not os.path.exists('./zMisc'):
        os.mkdir('zMisc')

    ncol = data_mean.shape[1]

    header = 'Prelim analysis to decide best strain definition to use\n[0]Stage'
    for k,name in enumerate(datalist):
        header+=',[{}]{}'.format(k+1,name)
    header+='[{}]Extenso_EpsY,[{}]Extenso_ey,[{}]BFC_EpsQ,[{}]BFC_eQ\n'.format(k+2,
            k+3,k+4,k+5)

    n.savetxt('zMisc/data_mean.dat', X=data_mean, 
            fmt='%.0f'+',%.6f'*(ncol-1),header=header+'Mean Values')
    n.savetxt('zMisc/data_med.dat', X=data_med, 
            fmt='%.0f'+',%.6f'*(ncol-1),header=header+'Median Values')

