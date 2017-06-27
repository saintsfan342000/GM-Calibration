import numpy as n
import matplotlib.pyplot as p
import figfun as f
p.style.use('mysty')

'''
Used to plot NEx vs NEx_alt, NExq vs gamma_alt for PT expts.
For deciding which definition to use in calibration

No difference in the NEx, NEy definitions (polyfit slope = 1)
Polyfit slope for gamma is ~1/2....but very noisy b/c shear value are so low
Recall from TT expts that the two gamma defs are virtually identical, but the technical stn defs are not.
'''

expts = n.array([2,3,4,7,8,10])

FS, SS = 30, 10
savefigs = 1 
savepath = '..'

# [0]Expt No., [1]Mon.Day, [2]Material, [3]Tube No., [4]Alpha, [5]Alpha-True    , [6]Mean Radius, [7]Thickness, [8]Eccentricity
key = n.genfromtxt('../../PTSummary.dat', delimiter=',')
# Sort by alpha
key = key[ n.in1d(key[:,0],expts) ]
key = n.flipud(key[ key[:,5].argsort() ])
expts = key[:,0].astype(int)

lines = [0]*len(expts)

fig1 = p.figure(figsize=(10,10))
ax1 = fig1.add_axes([.15,.15,.7,.7])
fig2 = p.figure(figsize=(10,10))
ax2 = fig2.add_axes([.15,.15,.7,.7])
fig3 = p.figure(figsize=(10,10))
ax3 = fig3.add_axes([.15,.15,.7,.7])

for k,X in enumerate(expts):
    
    relpath  = '../../../PT/GMPT-{}_FS{}SS{}'.format(X,FS,SS)
    expt, date, material, tube, alpha, a_true, Rm, thickness, ecc = key[ key[:,0] == X, : ].ravel()
    # Results.dat
    #   [0] Stage, [1,2,3]eps_x,q,xq(point avg), [4]eps_x(1"ext), [5]eps_q(BFcirc@mid), [6]d/L, Lg=4.330875
    # D = n.genfromtxt('{}/Results.dat'.format(relpath), delimiter=',')
    
    # [0]NEx [1]NEy [2]Gamma [3]F11-1 [4]F22-1 [5]atan(F12/F22) [6]epeq [7]Num pts
    D = n.genfromtxt('{}/loc_mean.dat'.format(relpath), delimiter=',')
    stg, time, F, P, sigx, sigq, LVDT, Disp = n.genfromtxt('{}/STPF.dat'.format(relpath), delimiter=',').T

    loc = sigq.argmax()
    D=D[:loc]

    
    # ax1:  Nex
    m, b = n.polyfit(D[:,0], D[:,3],1)
    label = '{:.0f} || {} || {:.2f}'.format(expt, alpha, m)
    lines[k] = ax1.plot(D[:,0], D[:,3], label=label)[0]
    if X == expts[-1]:
        ax1.set_xlabel('U-1 (Aramis)')
        ax1.set_ylabel('F11-1')
        ax1.set_title('Hoop Strain')
        f.ezlegend(ax1, title='Expt || $\\alpha$ || m')
            
    # ax2:  Ney
    m, b = n.polyfit(D[:,1], D[:,4],1)
    label = '{:.0f} || {} || {:.2f}'.format(expt, alpha, m)
    ax2.plot(D[:,1], D[:,4], label=label)
    if X == expts[-1]:
        ax2.set_xlabel('U-1 (Aramis)')
        ax2.set_ylabel('F22-1')
        ax2.set_title('Axial Strain')
        f.ezlegend(ax2, title='Expt || $\\alpha$ || m')

    # ax2:  Gamma
    m, b = n.polyfit(D[:,2], D[:,5],1)
    label = '{:.0f} || {} || {:.2f}'.format(expt, alpha, m)
    ax3.plot(D[:,2], D[:,5], label=label)
    if X == expts[-1]:
        ax3.set_xlabel('$\\gamma$: Aramis Def')
        ax3.set_ylabel('$\\gamma$\n(Halt)')
        ax3.set_title('Shear Stn')
        f.ezlegend(ax3, title='Expt || $\\alpha$ || m')

p.show('all')
