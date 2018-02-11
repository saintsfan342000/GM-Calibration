import numpy as n
from numpy import nanmean, nanstd
from scipy.integrate import cumtrapz
from scipy.io import loadmat
from scipy.interpolate import interp1d
from TrueStsStn import Iterate
import os
import matplotlib.pyplot as p
import figfun as f

p.style.use('mysty-quad')
p.rcParams['lines.markersize'] = 3
p.rcParams['axes.autolimit_mode'] = 'data'
p.rcParams['axes.xmargin'] = .01
p.rcParams['axes.ymargin'] = .01
p.rcParams['legend.fontsize'] = 9

'''
This script makes a 2x2 plot of the PT experiments, with dots at interpolated values of plastic
work.  It used to also do the Wp interpolation and save that file, but that has been merged into PTAnalysis1.  So this script only plots now.
'''

withuni = 0

# [0]Expt No., [1]Mon.Day, [2]Material, [3]Tube No., [4]Alpha, [5]Alpha-True, [6]Mean Radius, [7]Thickness, [8]Eccentricity
key = n.genfromtxt('../PTSummary.dat', delimiter=',')
key = key[ n.argsort(key[:,5]) ]
projects = key[:,0].astype(int)

for k,X in enumerate(projects):

    proj = 'GMPT-{}'.format(X)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==X].ravel()[4:8]
    os.chdir('../{}'.format(proj))

    D = n.genfromtxt('CalData.dat', delimiter=',')
    Dinttemp = n.genfromtxt('CalData_Interp.dat', delimiter=',')[::3]

    if k == 0:
        Wp_calcs = Dinttemp[:,1]
        Dint = n.empty((len(projects),*Dinttemp.shape))
        fig, ax1, ax2, ax3, ax4 = f.makequad()

    Dint[k] = Dinttemp
    # Truncate D to 100 psi beyond max Wp in Dint
    D = D[ D[:,1] <= Dinttemp[-1,1]+0.1 ]

    #ax1:  Axial True Sts-stn
    line, = ax1.plot(D[:,7], D[:,2], label='{} || {}'.format(X,alpha_true))
    MC, ML = line.get_color(), line.get_label()

    #ax2:  Hoop Stn-Stn
    ax2.plot(D[:,8], D[:,3], label=ML, color=MC)
    
    #ax3:  Axial vs hoop stn
    ax3.plot(D[:,8],D[:,7], label='{:.3f}'.format(Dint[k,0,12]), color=MC)

    #ax4:  Yield Surface
    # Plotting an expt at different Wps as markers of same color
    # Lines will be contours of Wp
    ax4.plot(Dint[k,:,3], Dint[k,:,2], 'o', color=MC)


# Done looping, now plot interpolated Wp levels
for k, X in enumerate(projects):
    proj = 'GMPT-{}'.format(X)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==X ].ravel()[4:8]
    os.chdir('../{}'.format(proj))
    
    #ax1:  Axial True Sts-stn
    if k == 0: cols = []
    for z,w in enumerate(Wp_calcs):
        if k == 0:
            line, = ax1.plot(Dint[k,z,7], Dint[k,z,2],'o')
            cols.append(line.get_color())
        else:
            ax1.plot(Dint[k,z,7], Dint[k,z,2],'o', color=cols[z])

    if X == projects[-1]:
        #ax1.axis(xmax=Dint[:,:,7].max()*1.1, xmin=Dint[:,:,7].min()*1.1)
        ax1.set_xlabel('$\\mathsf{e}_\\mathsf{x}^\\mathsf{p}$')
        ax1.set_ylabel('$\\tau_\\mathsf{x}$\n(ksi)')
        if withuni not in [1,2,3]:
            f.ezlegend(ax1, title='Exp || $\\alpha\\prime$', loc='lower right')
            f.myax(ax1)

    #ax2:  Hoop Stn-Stn
    for z,w in enumerate(Wp_calcs):
        ax2.plot(Dint[k,z,8], Dint[k,z,3],'o',color=cols[z])
    if X == projects[-1]:
        ax2.axis(xmax=Dint[:,:,8].max()*1.1)
        ax2.set_xlabel('$\\mathsf{e}_\\theta^\\mathsf{p}$')
        ax2.set_ylabel('$\\tau_\\theta$\n(ksi)')
        f.ezlegend(ax2, title='Exp || $\\alpha\\prime$', loc='lower right')
        f.myax(ax2)
   
    #ax3:  Axial vs hoop stn
    for z,w in enumerate(Wp_calcs):
        ax3.plot(Dint[k,z,8], Dint[k,z,7],'o',color=cols[z])
    if X == projects[-1]:
        xmin,xmax,ymin,ymax = n.array([Dint[:,:,8].min(), Dint[:,:,8].max(),
                        Dint[:,:,7].min(), Dint[:,:,8].max()])*1.1
        #ax3.axis(xmax=xmax,ymax=ymax,ymin=ymin)
        ax3.set_xlabel('$\\mathsf{e}_\\theta^\\mathsf{p}$')
        ax3.set_ylabel('$\\mathsf{e}_\\mathsf{x}^\\mathsf{p}$')
        if withuni not in [1,2,3]:
            f.ezlegend(ax3, 
                title='$\\mathsf{de}_\\mathsf{x}^\\mathsf{p}/\\mathsf{de}_\\theta^\\mathsf{p}$')
            f.myax(ax3)


    #ax4:  Yield surface
    for z,w in enumerate(Wp_calcs):
        if X == projects[-1] and withuni not in [1,2,3]:
            ax4.plot(Dint[:,z,3], Dint[:,z,2], color=cols[z],
                    label='{:.0f}'.format(w*1000), zorder=-10)
        else:
            DintPT = Dint.copy()

    if X == projects[-1]:
        #ax4.axis('equal')
        ax4.set_ylabel('$\\tau_\\mathsf{x}$\n(ksi)')
        ax4.set_xlabel('$\\tau_\\theta$ (ksi)')
        if withuni not in [1,2,3]:
            f.ezlegend(ax4, title='W$_\\mathsf{p}$ (psi)', loc='lower left')
            f.myax(ax4)

# if No uniaxial test
if withuni not in [1,2,3]:
    p.savefig('../PTCalibration.png', bbox_inches='tight', dpi=150)
else:
    
    # [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p, [5]eq_p, [6]ez_p
    D = n.genfromtxt('../Uniaxial/Uniax_6061_{}/CalData.dat'.format(withuni), delimiter=',')
    # [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p, [5]eq_p, [6]ez_p, [7]dexp/deqp, [8]dexp/deqp (moving)
    Dint = n.genfromtxt('../Uniaxial/Uniax_6061_{}/CalData_Interp.dat'.format(withuni), delimiter=',')[::3]
    D = D[ D[:,0] <= Dint[-1,0]+0.1 ]
    ax1.plot(D[:,4],D[:,1],'k',label='UT')
    for z,w in enumerate(Wp_calcs):
        ax1.plot(Dint[z,4], Dint[z,1],'o',color=cols[z])
    f.ezlegend(ax1, title='Exp || $\\alpha\\prime$', loc='lower right')
    f.myax(ax1)

    # Plotting uniax on sq-eq is worthless
    
    ax3.plot(D[:,5],D[:,4],'k',label='{:.3f}'.format(-1/Dint[2,7]))
    for z,w in enumerate(Wp_calcs):
        ax3.plot(Dint[z,5],Dint[z,4],'o',color=cols[z])
    f.ezlegend(ax3, 
                title='$\\mathsf{de}_\\mathsf{x}^\\mathsf{p}/\\mathsf{de}_\\theta^\\mathsf{p}$')
    f.myax(ax3)

    ax4.plot(Dint[:,1]*0, Dint[:,1], 'ko')
    for z,w in enumerate(Wp_calcs):
        sq = n.hstack((DintPT[:,z,3],0))
        sx = n.hstack((DintPT[:,z,2],Dint[z,1]))
        ax4.plot(sq,sx,color=cols[z],label='{:.0f}'.format(w*1000), zorder=-10)

    f.ezlegend(ax4, title='W$_\\mathsf{p}$ (psi)', loc='lower left')
    ax4.axis('equal')
    ax4.axis(xmin=0,ymin=0)
    f.myax(ax4)

    p.savefig('../UniPTData.png', bbox_inches='tight', dpi=150)


p.show('all')


