import numpy as n
import matplotlib.pyplot as p
import figfun as f
p.style.use('mysty')

'''
Used to plot NEx vs NEx_alt, NExq vs gamma_alt for TT expts.
For deciding which definition to use in calibration

Gamma definitions are very same (alt/aram >= 0.9)
NEx and NEy are >= 0.9 for alpha >= 2, but not a as close for < 2
This is expected, b/c we knew from eps v gamma plot in TT paper that
the Aramis definition is non-linear
'''

expts = n.array([1,2,3,4,5,7,9])

FS, SS = 19, 6
savefigs = 1 
savepath = '..'


# [0]Expt No., [1]Material, [2]Tube No., [3]Alpha, [4]Alpha-True, [5]Mean Radius, [6]Thickness, [7]Eccentricity
key = n.genfromtxt('../../TTSummary.dat', delimiter=',')
# Sort by alpha
key = key[ n.in1d(key[:,0],expts) ]
key = key[ key[:,3].argsort() ]
expts = key[:,0].astype(int)

lines = [0]*len(expts)

fig1 = p.figure(figsize=(10,10))
ax1 = fig1.add_axes([.15,.15,.7,.7])
fig2 = p.figure(figsize=(10,10))
ax2 = fig2.add_axes([.15,.15,.7,.7])
fig3 = p.figure(figsize=(10,10))
ax3 = fig3.add_axes([.15,.15,.7,.7])

for k,X in enumerate(expts):
    
    relpath  = '../../../TT/TTGM-{}_FS{}SS{}'.format(X,FS,SS)
    expt, material, tube, alpha, a_true, Rm, thickness, ecc = key[ key[:,0] == X, : ].ravel()
    # Results.dat
    #   [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg=4.330875
    # D = n.genfromtxt('{}/Results.dat'.format(relpath), delimiter=',')
    
    # [0]Stage [1]Time [2]NumPtsPassed [3]AxSts [4]ShSts [5]NEx [6]NEy [7]Gamma [8]F11-1 [9]F22-1 [10]atan(F12/F22) [11]epeq
    D = n.genfromtxt('{}/mean.dat'.format(relpath), delimiter=',')

    loc = n.genfromtxt('{}/prof_stages.dat'.format(relpath), delimiter=',', dtype=int)[1]

    D=D[:loc]

    
    # ax1:  Nex
    m, b = n.polyfit(D[:,5], D[:,8],1)
    label = '{:.0f} || {} || {:.2f}'.format(expt, alpha, m)
    lines[k] = ax1.plot(D[:,5], D[:,8], label=label)[0]
    if X == expts[-1]:
        ax1.set_xlabel('U-1 (Aramis)')
        ax1.set_ylabel('F11-1')
        ax1.set_title('Hoop Strain')
        f.ezlegend(ax1, title='Expt || $\\alpha$ || m')
            
    # ax2:  Ney
    m, b = n.polyfit(D[:,6], D[:,9],1)
    label = '{:.0f} || {} || {:.2f}'.format(expt, alpha, m)
    ax2.plot(D[:,6], D[:,9], label=label)
    if X == expts[-1]:
        ax2.set_xlabel('U-1 (Aramis)')
        ax2.set_ylabel('F22-1')
        ax2.set_title('Axial Strain')
        f.ezlegend(ax2, title='Expt || $\\alpha$ || m')

    # ax2:  Gamma
    m, b = n.polyfit(D[:,7], D[:,10],1)
    label = '{:.0f} || {} || {:.2f}'.format(expt, alpha, m)
    ax3.plot(D[:,7], D[:,10], label=label)
    if X == expts[-1]:
        ax3.set_xlabel('$\\gamma$: Aramis Def')
        ax3.set_ylabel('$\\gamma$\n(Halt)')
        ax3.set_title('Shear Stn')
        f.ezlegend(ax3, title='Expt || $\\alpha$ || m')

p.show('all')
