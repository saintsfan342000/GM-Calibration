import numpy as n
import matplotlib.pyplot as p
p.style.use('mysty')
import figfun as f
import os
'''
Investigate the difference between calculating log strains by
(1) n.log(U)
(2) n.log(eigU) then rotating back
(3) n.log(F22-1)
(4) n.log(Virst extensometer over /- 0.1")
'''

prefix = 'GMPT'
projects = [1,2,3,4,5,7,8,9]
projects = [8]

if prefix == 'TTGM':
    ht = .05 # the +/- height that defines my gagelen
    key = n.genfromtxt('../../TTSummary.dat', delimiter=',', usecols=(0,3))
elif prefix == 'GMPT':
    ht = .5
    key = n.genfromtxt('../../PTSummary.dat', delimiter=',', usecols=(0,4))
else:
    raise ValueError('Invalid prefix name given')

os.chdir('..') # Back out to AA_PyScripts

for expt in projects:
    proj = '{}-{}'.format(prefix,expt)
    print(proj)
    alpha = key[ key[:,0]==expt ].ravel()[1]

    os.chdir('../{}'.format(proj))

    (stage, NEx_aram,  NEy_aram, 
    NExy_aram,G_aram, NEx_alt, 
    NEy_alt,G_alt, ex_U, ey_U,ex_F,
    ey_F, ex_eigU, ey_eigU, Extenso_EpsY,
    Extenso_ey, BFC_EpsQ, BFC_eQ) = n.genfromtxt('./zMisc/data_mean.dat', delimiter=',', unpack=True)

    # [0]Stage,
    # [1] NEx_aram, [2] NEy_aram, [3]NExy_aram, [4] G_aram,
    # [5] NEx_alt, [6] NEy_alt, [7]G_alt, 
    # [8] ex_U, [9] ey_U,
    # [10] ex_F,[11] ey_F,
    # [12] ex_eigU, [13] ey_eigU
    # [14] Extenso_EpsY, [15] Extenso_ey, [16] BFC_EpsQ, [17] BFC_eQ


    # fig1 : Axial Strains (axial is y, hoop is x)
    fig1, ax11, ax12 = f.make21()
    fig1.suptitle('Axial Strain:  {}.  $\\alpha$ = {}'.format(proj,alpha))
    ax11.plot(stage, NEy_aram, label='Aram')
    ax11.plot(stage, NEy_alt, label='Haltom')
    ax11.plot([],[]) # Placeholder for color consistency
    ax11.plot(stage, Extenso_EpsY, label='Extenso')
    ax11.set_xlim(stage[0],stage[-1])
    f.eztext(ax11, 'Nominal', 'ul')
    f.ezlegend(ax11)
    ax12.plot(stage, ey_U, label='ln(U)')
    ax12.plot(stage, ey_F, label='ln(F)')
    ax12.plot(stage, ey_eigU, label='Q$^T$ln(EigU)Q')
    ax12.plot(stage, Extenso_ey, label='Extenso')
    ax12.set_xlim(stage[0],stage[-1])
    f.eztext(ax12, 'Logarithmic', 'ul')
    f.ezlegend(ax12)
    
    # fig2 : Hoop Strains (axial is y, hoop is x)
    fig2, ax21, ax22 = f.make21()
    fig2.suptitle('Hoop Strain:  {}.  $\\alpha$ = {}'.format(proj,alpha))
    ax21.plot(stage, NEx_aram, label='Aram')
    ax21.plot(stage, NEx_alt, label='Haltom')
    ax21.plot([],[]) # Placeholder
    ax21.plot(stage, BFC_EpsQ, label='BFC_Extens')
    ax21.set_xlim(stage[0],stage[-1])
    f.eztext(ax21, 'Nominal', 'ul')
    f.ezlegend(ax21)
    ax22.plot(stage, ex_U, label='ln(U)')
    ax22.plot(stage, ex_F, label='ln(F)')
    ax22.plot(stage, ex_eigU, label='Q$^T$ln(EigU)Q')
    ax22.plot(stage, BFC_eQ, label='BFC_Extens')
    ax22.set_xlim(stage[0],stage[-1])
    f.eztext(ax22, 'Logarithmic', 'ul')
    f.ezlegend(ax22)

    # fig3 : Strain ratio
    fig3, ax31, ax32 = f.make21()
    fig3.suptitle('Strain Ratio:  {}.  $\\alpha$ = {}'.format(proj,alpha))
    def poly_plot(ax,X,Y,label):
        m,b = n.polyfit(X,Y,1)
        ax.plot(X,Y,label='{}, m={:.3f}'.format(label,m))

    poly_plot(ax31, NEy_aram, NEx_aram, label='Aram')
    poly_plot(ax31, NEy_alt, NEx_alt, label='Haltom')
    ax31.plot([],[]) # Placeholder
    poly_plot(ax31, Extenso_EpsY, BFC_EpsQ, label='BFC_Extens')
    ax31.set_xlabel('$\\epsilon_\\mathsf{ax}$')
    ax31.set_ylabel('$\\epsilon_\\theta$')
    f.eztext(ax31, 'Nominal', 'ul')
    f.ezlegend(ax31, loc='lower right')
    poly_plot(ax32, ey_U, ex_U, label='ln(U)')
    poly_plot(ax32, ey_F, ex_F, label='ln(F)')
    poly_plot(ax32, ey_eigU, ex_eigU, label='Q$^T$ln(EigU)Q')
    poly_plot(ax32, Extenso_ey, BFC_eQ, label='BFC_Extens')
    ax32.set_xlabel('$\\epsilon_\\mathsf{ax}$')
    ax32.set_ylabel('$\\epsilon_\\theta$')
    f.eztext(ax32, 'Logarithmic', 'ul')
    f.ezlegend(ax32, loc='lower right')

p.show()
