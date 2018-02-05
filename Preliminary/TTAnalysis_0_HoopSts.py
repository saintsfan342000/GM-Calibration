import numpy as n
import matplotlib.pyplot as p
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import os

'''
From my Abaqus analyses, I determined that the hoop stress is *approximately*
linearly related to the axial stress at a given level of plastic work.

This script reads those abaqus jobs, computes the linear fit parameters, and saves
them to  Calibration/zMisc/HoopAxial.dat
'''

expts = [2, 3, 5, 7]
constit = 'h8'
Wp_calcs = n.arange(0.1,2.100,.100)

# [Ax0] Expt, [Ax1]Wp_calc, [Ax2]StsComponents
S_int = n.empty((len(expts), len(Wp_calcs), 6))
curdir = os.getcwd()
os.chdir('../../../../../Abaqus/TT/Results/HoopSts')

for k,x in enumerate(expts):
    #[0]Nom AxSts, [1]Nom Shear Sts, [2]d/Lg lo, [3]Rot lo, [4]d/Lg hi, [5]Rot hi, [6]d/Lg new, [7]Rot new
    sim = n.genfromtxt('{}{}/disprot.dat'.format(x,constit), delimiter=',')
    simloc = sim[:,1].argmax()
    #[0]Srr, [1]Sqq, [2]Szz, [4]Srq, [5]Srz?, [6]Sqz
    S = n.genfromtxt('{}{}/S_anal_zone.dat'.format(x,constit), delimiter=',')
    
    PE = n.genfromtxt('{}{}/EqStsStn_anal_zone.dat'.format(x,constit), delimiter=',')
    Wp = cumtrapz(PE[:,0], PE[:,1], initial=0)

    S_int[k] = interp1d(Wp, S, axis=0).__call__(Wp_calcs)

MB = n.empty((len(Wp_calcs),3))
for k,w in enumerate(Wp_calcs*1000):
    x = S_int[:,k,2]
    y = S_int[:,k,1]
    MB[k] = w, *n.polyfit(x,y,1)


os.chdir(curdir)
header = 'Relationship between Hoop and Axial Stress\n'
header += 'SigQ = m*SigX + b\n'
header += '[0]Wp (psi), [1]m, [2]b'
n.savetxt('../../zMisc/HoopAxial.dat', X=MB,
        header=header, fmt='%.0f, %.8f, %.8f')

