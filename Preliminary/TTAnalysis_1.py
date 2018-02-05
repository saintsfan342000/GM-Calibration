import numpy as n
pi=n.pi
from numpy import nanmean, nanstd
from numpy.linalg import eigvalsh, eigh
from scipy.integrate import cumtrapz
from scipy.io import loadmat
from scipy.interpolate import interp1d
from mysqrtm import mysqrtm
import os

'''
This script is very exploratoy.

For TT Experiments
(1)Log strains
(2)True Stresses
(3)Log plastic strains
(4)Plastic work

In this script, the hoop stress and strain are not used in calculating the plastic work.
The hoop stress is needed, though, for the Barlat equivalent stress.
Thus this script gives a "first estimate" of the plastic work.

To calculate the plastic axial strain, I neglect the hoop contribution. i.e.,
ex_p = ex - (sig_x - nu*sig_q)/E , but sig_q is neglected all together.

In addition, I calculate the tru thickness in two different ways:
    (1) Calculating principle stretches from U tensor and assuming incompressibility
    (2) Forming my own strain tensor = [[F00-1, G/2], [G/2, F11-1]] and getting the principle
        strains from there, then assume incompressibility

'''

prefix = 'TTGM'
ht = .05
E = 9750
nu = 0.319
makeplots = 0
if makeplots:
    import matplotlib.pyplot as p
    import figfun as f
    p.style.use('mysty-quad')
    p.rcParams['lines.markersize'] = 4
    p.rcParams['axes.autolimit_mode'] = 'data'
    p.rcParams['axes.xmargin'] = .02
    p.rcParams['axes.ymargin'] = .02

# [0]Expt No., [1]Material, [2]Tube No., [3]Alpha, [4]Alpha-True,
# [5]Mean Radius, [6]Thickness, [7]Eccentricity
key = n.genfromtxt('../TTSummary.dat', delimiter=',')
key = key[ n.argsort(key[:,4]) ]
projects = key[:,0].astype(int)

for expt in projects:

    proj = '{}-{}'.format(prefix,expt)
    print('\n'+proj)
    alpha, alpha_true, Rm, thickness = key[ key[:,0]==expt ].ravel()[4:8]

    os.chdir('../{}'.format(proj))

   # [0]Stg [1]Time [2]AxSts [3]ShSts [4]AxForce [5]Torque [6]MTS Disp [7]MTS Rot 
    stf = n.genfromtxt('STF.dat', delimiter=',')
    force, torque = stf[:,4:6].T
    prof_stgs = n.genfromtxt('prof_stages.dat', delimiter=',', dtype=int)
    stages, Force, P, sig_x, sig_q = stf[:,[0,2,3,4,5]].T
    stages = stages.astype(int)

    # [0]Stage, [1]Wp, [2]SigX_Tru, [3]Tau_True, 
    # [4]ex, [5]eq, [6]exq, [7]e3
    # [8]ep_x, [9]ep_q, [1]exq_p, [11]ep_3
    # [10]R_tru, [11]th_tru
    D = n.empty((len(stages),12))
    
    #Progressbar (10 # long)
    pbarstages = n.linspace(stages[-1],0,20,dtype=int)

    #Cycle through the stages
    print('#'*20)
    for stage in stages[:0:-1]
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
        gamma = n.abs(n.arctan(F[:,0,1]/F[:,1,1]))
        rat = gamma / (F[:,1,1]-1)
        ratmean = nanmean(rat)
        ratdev = nanstd(rat)
        keeps = (rat>=ratmean-1.5*ratdev) & (rat<=ratmean+1.5*ratdev)
        F = F.compress(keeps, axis=0)
        # Filter again
        gamma = n.abs(n.arctan(F[:,0,1]/F[:,1,1]))
        rat = gamma / (F[:,1,1]-1)
        ratmean = nanmean(rat)
        ratdev = nanstd(rat)
        keeps = (rat>=ratmean-0.5*ratdev) & (rat<=ratmean+0.5*ratdev)
        F = F.compress(keeps, axis=0)
        
        # First method of strain and true thickness
        # Use the aramis stretch tensor
        FtF = n.einsum('...ji,...jk',F,F)
        U = mysqrtm( FtF )     #Explicit calculation of sqrtm
        U = U.mean(axis=0)
        Rtru = R*U[0,0]
        NE = U - n.eye(2)
        (eps1,eps2), Q = eigh(NE)
        e1,e2 = n.log(eps1+1), n.log(eps2+1)
        Elog_prin = n.array([[e1,0],[0,e2]]) 
        Elog = (Q.T)@(Elog_prin)@(Q)
        eq, ex = Elog.diagonal()
        exq = n.abs(Elog[0,1])
        G = n.abs(n.arctan(U[0,1]/(U[0,0])) + n.arctan(U[0,1]/(U[1,1])))
        L1,L2 = eigvalsh(U)
        th_tru = thickness/(L1*L2)
        sig_x = force/(2*pi*Rtru*th_tru)
        sig_q = sig_x*.12 + 5 # Based on my analysis polyfit
        tau_xq = torque/(2*pi*Rtru**2*th_tru)
        ep_x = ex - (sig_x-nu*sig_q)/E
        ep_q = eq - (sig_q-nu*sig_x)/E
        ep_xq = exq - (1+nu)tau_xq/E
        Gp = G - 2*(1+nu)*tau_xq/E
        Wp1 = n.trapz(sig_x,ep_x) + n.trapz(tau_xq,Gp)
        Wp2 = n.trapz(sig_x,ep_x) + 2*n.trapz(tau_xq,ep_xq)
        Wp3 = n.trapz(sig_x,ep_x) + n.trapz(tau_xq,Gp) + n.trapz(sig_q,ep_q)

        # [0]Stage, [1]Wp, [2]SigX_Tru, [3]SigQ_True [4]Tau_XQ_Tru, 
        # [4]ex, [5]eq, [6]2*exq, [7]G,
        # [8]ep_x, [9]ep_q, [9]ep_3, [10]R_tru, [11]th_tru 
        D[k] = (stage, Wp,
                sig_x, sig_q, tau_xq, 
                ex, eq, 2*exy, G,
                ep_x, ep_q, 2*ep_exy, Gp




        # Haltom strain definitions
        epsx = (F[:,1,1] - 1).mean()
        epsq = (F[:,0,0] - 1).mean()
        G =  n.arctan(F[:,0,1]/F[:,1,1]).mean()

        # Make my own nominal strain tensor since I decided
        # this was the best way
        Enom = n.array([[epsq, G/2],[G/2,epsx]]);
        (eps1,eps2), Q = eigh(Enom)
        e1,e2 = n.log(eps1+1), n.log(eps2+1)
        Elog_prin = n.array([[e1,0],[0,e2]]) 
        Elog = (Q.T)@(Elog_prin)@(Q)

        Rtru = Rm*(1+epsq)

        # Second method of true thickness
        # Form my own log strain tensor
        U = n.array([[eq,G/2],[G/2,ex]])
        L1,L2 = n.exp(eigvalsh(U))
        th_tru2 = thickness/(L1*L2)
        sig_x2 = force/(2*pi*Rtru*th_tru2)
        tau_xq2 = torque/(2*pi*Rtru**2*th_tru2)
        ep_x2 = ex - sig_x2/E
        Gp2 = G - 2*(1+nu)*tau_xq2/E
        Wp2 = n.trapz(sig_x2,ep_x2) + n.trapz(tau_xq2,Gp2)

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









    



       


