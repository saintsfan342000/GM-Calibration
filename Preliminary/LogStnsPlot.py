import numpy as n
import matplotlib.pyplot as p
import figfun as f
import os
from cycler import cycler

#cols = [k['color'] for k in list(p.rcParams['axes.prop_cycle'])]
#LS = ['-', '--', ':', '-.']
#p.rcParams['axes.prop_cycle'] = cycler('color', cols) * cycler('linestyle', LS)

'''
Investigate the difference between calculating log strains by
(1) n.log(U)
(2) n.log(eigU) then rotating back
(3) n.log(F22-1)
(4) n.log(Virst extensometer over /- 0.1")
'''

prefix = 'GMPT'
savefigs = True

pwd = os.getcwd()

if prefix == 'TTGM':
    ht = .05 # the +/- height that defines my gagelen
    key = n.genfromtxt('../../TTSummary.dat', delimiter=',', usecols=(0,3))
    projects = key[:,0].astype(int)
elif prefix == 'GMPT':
    ht = .5
    key = n.genfromtxt('../../PTSummary.dat', delimiter=',', usecols=(0,4))
    projects = key[:,0].astype(int)
else:
    raise ValueError('Invalid prefix name given')

os.chdir('..') # Back out to AA_PyScripts

def poly_plot(ax,X,Y,label):
    m,b = n.polyfit(X,Y,1)
    ax.plot(X,Y,label='{}\nm={:.3f}'.format(label,m))

for expt in projects:
    proj = '{}-{}'.format(prefix,expt)
    print(proj)
    alpha = key[ key[:,0]==expt ].ravel()[1]

    os.chdir('../{}'.format(proj))

    D = n.genfromtxt('./zMisc/data_mean.dat', delimiter=',', unpack=True)

    if n.any(n.isnan(D)):  print('Some nans in the data')

    (stage, NEx_aram,  NEy_aram, NExy_aram,G_aram,
    NEx_RUR, NEy_RUR, NExy_RUR, G_RUR,
    NEx_halt, NEy_halt, G_halt,
    ex_U, ey_U,
    ex_RUR, ey_RUR,
    ex_F,ey_F,
    ex_eigU, ey_eigU,
    Extenso_EpsY,Extenso_ey, BFC_EpsQ,
    BFC_eQ) = D

    NExy_aram,G_aram,NExy_RUR,G_RUR,G_halt = map(n.abs, (NExy_aram,G_aram,NExy_RUR,G_RUR,G_halt))

    p.style.use('mysty-12')
    # fig1 : Axial Strains (axial is y, hoop is x)
    fig1, ax11, ax12 = f.make21()
    ax11.set_title('Axial Strain:  {}.  $\\alpha$ = {}'.format(proj,alpha))
    ax11.plot(stage, NEy_aram, label='Aram')
    ax11.plot(stage, NEy_halt, label='Halt')
    ax11.plot(stage, NEy_RUR, label='RUR')
    ax11.plot([],[]) # Placeholder for color consistency
    ax11.plot(stage, Extenso_EpsY, label='Extenso')
    ax11.set_xlim(stage[0],stage[-1])
    ax11.set_ylabel('$\\epsilon_\\mathsf{ax}$')
    f.eztext(ax11, 'Nominal', 'ul')
    f.ezlegend(ax11)
    f.myax(ax11)
    ax12.plot(stage, ey_U, label='ln(U)')
    ax12.plot(stage, ey_F, label='ln(F)')
    ax12.plot(stage, ey_RUR, label='RUR')
    ax12.plot(stage, ey_eigU, label='Q$^T$ln(EigU)Q')
    ax12.plot(stage, Extenso_ey, label='Extenso')
    ax12.set_xlim(stage[0],stage[-1])
    ax12.set_ylabel('e$_\\mathsf{ax}$')
    f.eztext(ax12, 'Logarithmic', 'ul')
    f.ezlegend(ax12)
    f.myax(ax12)
    
    # fig2 : Hoop Strains (axial is y, hoop is x)
    fig2, ax21, ax22 = f.make21()
    ax21.set_title('Hoop Strain:  {}.  $\\alpha$ = {}'.format(proj,alpha))
    ax21.plot(stage, NEx_aram, label='Aram')
    ax21.plot(stage, NEx_halt, label='Halt')
    ax21.plot(stage, NEx_RUR, label='RUR')
    ax21.plot([],[]) # Placeholder
    ax21.plot(stage, BFC_EpsQ, label='BFC_Extens')
    ax21.set_xlim(stage[0],stage[-1])
    ax21.set_ylabel('$\\epsilon_\\theta$')
    f.eztext(ax21, 'Nominal', 'ul')
    f.ezlegend(ax21)
    f.myax(ax21)
    ax22.plot(stage, ex_U, label='ln(U)')
    ax22.plot(stage, ex_F, label='ln(F)')
    ax22.plot(stage, ex_RUR, label='RUR')
    ax22.plot(stage, ex_eigU, label='Q$^T$ln(EigU)Q')
    ax22.plot(stage, BFC_eQ, label='BFC_Extens')
    ax22.set_xlim(stage[0],stage[-1])
    ax22.set_ylabel('e$_\\theta$')
    f.eztext(ax22, 'Logarithmic', 'ul')
    f.ezlegend(ax22)
    f.myax(ax22)

    # fig3 : Strain ratio
    fig3, ax31, ax32 = f.make21()
    ax31.set_title('Strain Ratio:  {}.  $\\alpha$ = {}'.format(proj,alpha))
    poly_plot(ax31, NEy_aram, NEx_aram, label='Aram')
    poly_plot(ax31, NEy_halt, NEx_halt, label='Halt')
    poly_plot(ax31, NEy_RUR, NEx_RUR, label='RUR')
    ax31.plot([],[]) # Placeholder
    poly_plot(ax31, Extenso_EpsY, BFC_EpsQ, label='BFC_Extens')
    ax31.set_xlabel('$\\epsilon_\\mathsf{ax}$')
    ax31.set_ylabel('$\\epsilon_\\theta$')
    f.eztext(ax31, 'Nominal', 'ul')
    f.ezlegend(ax31)
    f.myax(ax31)
    poly_plot(ax32, ey_U, ex_U, label='ln(U)')
    poly_plot(ax32, ey_F, ex_F, label='ln(F)')
    poly_plot(ax32, ey_RUR, ex_RUR, label='RUR')
    poly_plot(ax32, ey_eigU, ex_eigU, label='Q$^T$ln(EigU)Q')
    poly_plot(ax32, Extenso_ey, BFC_eQ, label='BFC_Extens')
    ax32.set_xlabel('e$_\\mathsf{ax}$')
    ax32.set_ylabel('e$_\\theta$')
    f.eztext(ax32, 'Logarithmic', 'ul')
    f.ezlegend(ax32)
    f.myax(ax32)

    # fig4: Gamma, only if TT
    fig4 = None
    if prefix == 'TTGM':
        fig4,ax41,ax42 = f.make12()
        fig4.suptitle('Shear Strain:{}.  $\\alpha$ = {}'.format(proj,alpha))
        ax41.plot(stage,G_aram,label='Aram')
        ax41.plot(stage,G_halt,label='Halt')
        ax41.plot(stage,G_RUR,label='RUR')
        ax41.set_ylabel('$\\gamma$')
        f.ezlegend(ax41)
        f.myax(ax41)

        poly_plot(ax42, NExy_aram, G_aram, 'Aram')
        poly_plot(ax42, NExy_aram, G_halt, 'Halt')
        poly_plot(ax42, NExy_RUR, G_RUR, 'RUR')
        ax42.set_xlabel('$\\epsilon_\\mathsf{xy}$')
        ax42.set_ylabel('$\\gamma$')
        f.ezlegend(ax42)
        f.myax(ax42)



    if savefigs:
        path = '{}/LogStnPlots/{}'.format(pwd, prefix)
        figs = [fig1,fig2,fig3]
        titles = ['AxStn','HoopStn','StnRatio']
        if prefix == 'TTGM':
            figs.append(fig4)
            titles.append('ShearStn')
        for fig,title in zip(figs,titles):
            # Save to Prelim folder
            fig.savefig('{}/{}_{}_{}.png'.format(path,title,expt,alpha),
                        dpi=100, bbox_inches='tight')
            # Save to local misc folder
            fig.savefig('{}/{}_{}_{}.png'.format('zMisc',title,expt,alpha),
                        dpi=100, bbox_inches='tight')
        p.close('all')
    else:
        p.show('all')

