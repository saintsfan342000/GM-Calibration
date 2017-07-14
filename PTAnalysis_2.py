import numpy as n
from numpy import nanmean, nanstd
from scipy.integrate import cumtrapz
from scipy.io import loadmat
from scipy.interpolate import interp1d
from TrueStsStn import Iterate
import os

'''
This script makes a 2x2 plot of the PT experiments, with dots at interpolated values of plastic
work.  More importantly, it also calculates the strain ratio and stress state at these
values of plastic work and save the file CalData_Interp.dat
'''

prefix = 'GMPT'
ht = .5
E = 9750
nu = 0.319
makeplots = 1
if makeplots:
    import matplotlib.pyplot as p
    import figfun as f
    p.style.use('mysty-quad')
    p.rcParams['lines.markersize'] = 4
    p.rcParams['axes.autolimit_mode'] = 'data'
    p.rcParams['axes.xmargin'] = .01
    p.rcParams['axes.ymargin'] = .01
    p.rcParams['legend.fontsize'] = 9


# [0]Expt No., [1]Mon.Day, [2]Material, [3]Tube No., [4]Alpha, [5]Alpha-True, [6]Mean Radius, [7]Thickness, [8]Eccentricity
key = n.genfromtxt('../PTSummary.dat', delimiter=',')
key = key[ n.argsort(key[:,5]) ]
projects = key[:,0].astype(int)


Wp_calcs = n.arange(200,1200,200)/1000

for k,X in enumerate(projects):

    proj = '{}-{}'.format(prefix,X)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==X].ravel()[4:8]
    os.chdir('../{}'.format(proj))

    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, 
    # [6]e3, [7]ep_x, [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru
    D = n.genfromtxt('CalData.dat', delimiter=',')


    Dinttemp = interp1d(D[:,1],D,axis=0).__call__(Wp_calcs)
    if k == 0:
        Dint = n.empty((len(projects),len(Wp_calcs),Dinttemp.shape[1]))
        fig, ax1, ax2, ax3, ax4 = f.makequad()

    Dint[k] = Dinttemp

    #ax1:  Axial True Sts-stn
    line, = ax1.plot(D[:,7], D[:,2], label='{} || {}'.format(X,alpha_true))
    MC, ML = line.get_color(), line.get_label()

    #ax2:  Hoop Stn-Stn
    ax2.plot(D[:,8], D[:,3], label=ML, color=MC)
    
    #ax3:  Axial vs hoop stn
    rng = (D[:,1]>=Wp_calcs[0]) & (D[:,1]<=Wp_calcs[-1])
    m,b = n.polyfit(D[rng,8],D[rng,7],1)
    ax3.plot(D[:,8],D[:,7], label='{:.3f}'.format(m), color=MC)

    #ax4:  Yield Surface
    ax4.plot(Dint[k,:,3], Dint[k,:,2], 'o', ms=4, color=MC)


# Done looping, now plot interpolated Wp levels
for k, X in enumerate(projects):
    proj = '{}-{}'.format(prefix,X)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==X ].ravel()[4:8]
    os.chdir('../{}'.format(proj))
    
    #ax1:  Axial True Sts-stn
    if k == 0: cols = []
    for z,w in enumerate(Wp_calcs):
        line, = ax1.plot(Dint[k,z,7], Dint[k,z,2],'o',ms=4)
        if k == 0:
            cols.append(line.get_color())
    if X == projects[-1]:
        ax1.axis(xmax=Dint[:,:,7].max()*1.1, xmin=Dint[:,:,7].min()*1.1)
        ax1.set_xlabel('$\\mathsf{e}_\\mathsf{x}^\\mathsf{p}$')
        ax1.set_ylabel('$\\tau_\\mathsf{x}$\n(ksi)')
        f.ezlegend(ax1, title='Exp || $\\alpha\\prime$', loc='lower right')
        f.myax(ax1)

    #ax2:  Hoop Stn-Stn
    for z,w in enumerate(Wp_calcs):
        ax2.plot(Dint[k,z,8], Dint[k,z,3],'o',ms=4, color=cols[z])
    if X == projects[-1]:
        ax2.axis(xmax=Dint[:,:,8].max()*1.1)
        ax2.set_xlabel('$\\mathsf{e}_\\theta^\\mathsf{p}$')
        ax2.set_ylabel('$\\tau_\\theta$\n(ksi)')
        f.ezlegend(ax2, title='Exp || $\\alpha\\prime$', loc='lower right')
        f.myax(ax2)
   
    #ax3:  Axial vs hoop stn
    for z,w in enumerate(Wp_calcs):
        ax3.plot(Dint[k,z,8], Dint[k,z,7],'o',ms=4, color=cols[z])
    if X == projects[-1]:
        xmin,xmax,ymin,ymax = n.array([Dint[:,:,8].min(), Dint[:,:,8].max(),
                        Dint[:,:,7].min(), Dint[:,:,8].max()])*1.1
        ax3.axis(xmax=xmax,ymax=ymax,ymin=ymin)
        ax3.set_xlabel('$\\mathsf{e}_\\theta^\\mathsf{p}$')
        ax3.set_ylabel('$\\mathsf{e}_\\mathsf{x}^\\mathsf{p}$')
        f.ezlegend(ax3, 
                title='$\\mathsf{de}_\\mathsf{x}^\\mathsf{p}/\\mathsf{de}_\\theta^\\mathsf{p}$')
        f.myax(ax3)


    #ax4:  Yield surface
    for z,w in enumerate(Wp_calcs):
        if X == projects[-1]:
            ax4.plot(Dint[:,z,3], Dint[:,z,2], color=cols[z],
                    label='{:.0f}'.format(w*1000), zorder=-10)
        else:
            ax4.plot(Dint[:,z,3], Dint[:,z,2], color=cols[z], zorder=-10)
    if X == projects[-1]:
        #ax4.axis('equal')
        ax4.set_ylabel('$\\tau_\\mathsf{x}$\n(ksi)')
        ax4.set_xlabel('$\\tau_\\theta$\n(ksi)')
        f.ezlegend(ax4, title='W$_\\mathsf{p}$ (psi)', loc='lower left')
        f.myax(ax4)

p.savefig('../PTCalibration.png', bbox_inches='tight', dpi=150)
p.show('all')


Wp_calcs = n.arange(200,1100,100)/1000

for k,X in enumerate(projects):

    proj = '{}-{}'.format(prefix,X)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==X].ravel()[4:8]
    os.chdir('../{}'.format(proj))

    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, 
    # [6]e3, [7]ep_x, [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru
    D = n.genfromtxt('CalData.dat', delimiter=',')

    Dint = interp1d(D[:,1],D,axis=0).__call__(Wp_calcs)

    rng = (D[:,1]>=Wp_calcs[0]) & (D[:,1]<=Wp_calcs[-1])

    m,b = n.polyfit(D[rng,8],D[rng,7],1)

    erange = n.array([m]*Dint.shape[0])

    Dint = n.c_[Dint,erange]

    header='[0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru'
    
    header='[0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru, [12]dexp/deqp'
    n.savetxt('CalData_Interp.dat', X=Dint, fmt='%.3f'+', %.6f'*12, header=header)

