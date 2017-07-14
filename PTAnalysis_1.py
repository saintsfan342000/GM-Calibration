import numpy as n
from numpy import nanmean, nanstd
from scipy.integrate import cumtrapz
from scipy.io import loadmat
from scipy.interpolate import interp1d
from TrueStsStn import Iterate
import os

'''
For PT Experiments
(1)Log strains
(2)True Stresses
(3)Log plastic strains
(4)Plastic work
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
    p.rcParams['axes.xmargin'] = .02
    p.rcParams['axes.ymargin'] = .02


# [0]Expt No., [1]Mon.Day, [2]Material, [3]Tube No., [4]Alpha, [5]Alpha-True, [6]Mean Radius, [7]Thickness, [8]Eccentricity
key = n.genfromtxt('../PTSummary.dat', delimiter=',')
projects = key[:,0].astype(int)

#projects = [2]

for expt in projects:

    proj = '\n{}-{}'.format(prefix,expt)
    print(proj)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==expt ].ravel()[4:8]

    os.chdir('../{}'.format(proj))

    # [0]Stage, [1]Time, [2]Force(kip), [3]Pressure(ksi), [4]NomAxSts(ksi), [5]NomHoopSts(ksi), [6]LVDT(volt), [7]MTSDisp(in)
    stf = n.genfromtxt('STPF.dat', delimiter=',')
    prof_stgs = n.genfromtxt('prof_stages.dat', delimiter=',', dtype=int)
    stages, Force, P, sig_x, sig_q = stf[:,[0,2,3,4,5]].T
    stages = stages.astype(int)

    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru
    D = n.empty((len(stages),12))
    
    #Progressbar (10 # long)
    pbarstages = n.linspace(stages[-1],0,20,dtype=int)

    #Cycle through the stages
    print('#'*20)
    for stage in stages[:0:-1]:
        if stage in pbarstages:
            print('#', end='', flush=True)
        # Load up the stage
        name = 'stage_{}'.format(stage)
        A = loadmat('{}'.format(proj), variable_names=name)[name]
        #[0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)

        if stage == stages[-1]:
            Q = n.arctan2(A[:,2], A[:,4])*180/n.pi
            q_rng = Q.max()-Q.min()
            q_mid = Q.min()+q_rng/2
         
        Q = n.arctan2(A[:,2], A[:,4])*180/n.pi
        rng = (n.abs(A[:,3])<=ht) & (Q>=q_mid-q_rng/4) & (Q<=q_mid+q_rng/4)
        F=A[rng,-4:].reshape(-1,2,2)   # A "stack" of 2x2 deformation gradients 

        # Filter
        rat = (F[:,0,0]-1) / (F[:,1,1]-1)
        ratmean = nanmean(rat)
        ratdev = nanstd(rat)
        keeps = (rat>=ratmean-1.5*ratdev) & (rat<=ratmean+1.5*ratdev)
        F = F.compress(keeps, axis=0)
        # Filter again
        rat = (F[:,0,0]-1) / (F[:,1,1]-1)
        ratmean = nanmean(rat)
        ratdev = nanstd(rat)
        keeps = (rat>=ratmean-0.5*ratdev) & (rat<=ratmean+0.5*ratdev)
        F = F.compress(keeps, axis=0)
        
        # Haltom strain definitions
        ex = n.log(F[:,1,1]).mean()
        eq = n.log(F[:,0,0]).mean()
        G = n.arctan(F[:,0,1]/F[:,1,1]).mean()
        Rtru = Rm*(1+eq)
        
        (th_tru, tau_x, tau_q, 
         ep_x, ep_q, ep_3, e3) = Iterate(P[stage], Force[stage], thickness, Rtru, ex, eq, E, nu)

        Wp = tau_x*ep_x + tau_q*ep_q
        
        D[stage,0] = stage
        D[stage, 2:] = tau_x, tau_q, ex, eq, e3, ep_x, ep_q, ep_3, Rtru, th_tru

        ### End iteration through stage
    D[0] = 0
    D[0,[2,3,10,11]] = sig_x[0], sig_q[0], Rm, thickness

    # nancheck
    locs = n.where(n.any(n.isnan(D),axis=1))[0]
    if len(locs)>=1:
        print('\nWarning! nans found in stages' + (' {}'*len(locs)).format(*locs))
        # Replace nan rows with preceding
        D[locs,1:] = D[locs-1,1:]

    D[1:,1] = cumtrapz(D[:,2],D[:,7]) + cumtrapz(D[:,3], D[:,8])
    
    header='[0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True, [4]ex, [5]eq, [6]e3, [7]ep_x, [8]ep_q, [9]ep_3, [10]R_tru, [11]th_tru'
    fmt='%.0f'+', %.6f'*11
    n.savetxt('CalData.dat', X=D, fmt=fmt,header=header)

    if makeplots == True:
        eps_x = n.exp(D[:,4])-1
        eps_q = n.exp(D[:,5])-1
        fig, ax1, ax4, ax2, ax3 = f.makequad()

        ax4.axis('off')
        ax4.text(.5,.5,'{}-{}.  $\\alpha$ = {}'.format(prefix, expt, alpha_true),
                ha='center',transform=ax4.transAxes)

        Wp = n.linspace(D[prof_stgs[0],1], D[prof_stgs[2],1], 8)
        Dint = interp1d(D[:,1], D, axis=0).__call__(Wp)
        
        #ax1:  True Sts-stn
        line, = ax1.plot(D[:,4], D[:,2])
        lc,  label = line.get_color(), '$\\sigma_x-\\mathsf{e}_\\mathsf{x}$\n'
        f.eztext(ax1, label, 'br', color=lc)
        line, = ax1.plot(D[:,5], D[:,3])
        lc,  label = line.get_color(), '$\\sigma_\\theta-\\mathsf{e}_\\theta$'
        f.eztext(ax1, label, 'br', color=lc)
        cols = []
        for C,X in enumerate(Dint):
            cols.append(ax1.plot(X[4],X[2],'o',label='{:.0f} psi'.format(Wp[C]*1000))[0])
        for C,X in enumerate(Dint):
            ax1.plot(X[5],X[3],'o', mfc=cols[C].get_mfc(), mec=cols[C].get_mec())
        leg2 = f.ezlegend(ax1, markers=True)
        ax1.set_xlabel('e')
        ax1.set_ylabel('$\\tau$\n(ksi)')
        ax1.axis(xmin=min(D[:,4].min(), D[:,5].min()))
        f.ezlegend(ax1, markers=True, title='W$_\\mathsf{p}$ (psi)')
        f.myax(ax1)

        #ax2:  Stn-Stn
        ax2.plot(D[:,8]*100, D[:,7]*100)
        for C,X in enumerate(Dint):
            ax2.plot(X[8]*100,X[7]*100,'o', mfc=cols[C].get_mfc(),
                     mec=cols[C].get_mec(),label='W$_p$={:.1f}'.format(Wp[C]))
        rng = (D[:,1]>=Wp[0]) & (D[:,1]<=Wp[-1])
        m,b = n.polyfit(D[rng,8]*100, 100*D[rng,7], 1)
        x = n.array([ D[rng,8].min(), D[rng,8].max() ])*100
        y = m*x+b
        ax2.plot(x,y,'k-', zorder=-5)
        f.eztext(ax2, '$\\frac{de_x^p}{de_\\theta^p}=$'+'{:.3f}'.format(m), 'br')
        ax2.axis(xmin=D[:,8].min())
        ax2.set_ylabel('$\\mathsf{e}_\\mathsf{x}$\n(%)')
        ax2.set_xlabel('$\\mathsf{e}_\\mathsf{\\theta}$ (%)')


        f.myax(ax2)

        #ax3:  Plastic work vs time
        ax3.plot(stf[:,1], D[:,1])
        ax3.axis(ymin=D[:,1].min())
        ax3.set_xlabel('t (s)')
        ax3.set_ylabel('W$_\\mathsf{p}$\n(ksi)')
        f.myax(ax3)

        p.savefig('CalData.png', bbox_inches='tight', dpi=100)
        p.close('all')









    



       


