import numpy as n
import matplotlib.pyplot as p
from pandas import read_csv
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import figfun as f
p.style.use('mysty-sub')

'''
This gets the log plastic strains, stresses, and ratios 
from the Uniaxial experiment #3.  Expt3 I did on the flat speciment
using DIC.  In the Uniaxial PyFiles folder there is a Jupyter notebook
where I calculate the strain ratios from Expts 1 and 2, in which I used 
strain gages, and also expt 3.  The e_trp/e_axp ratio was shown to be very similar 
between expts all the experiments.  So I've decided to use expt 3 here since
the SGs popped off expts 1 and 2 at Wp levels of ~1000 and 400 psi, rectively.
By using the DIC expt I can use the data up to an arbitrarily large Wp
'''

x = 1
mat=6061
# Only expts 1 and 2 used SGs and need an sgcut
sgcut = ['worthless', 0.03, 0.014, 'worthless']
key = n.genfromtxt('../Uniaxial/ExptSummary.dat',delimiter=',')

area, gotime, E, ν = key[ (key[:,0] == x) & (key[:,1]==mat), 3:7].ravel()
# gotime is the time in the labview file where the test starts

if x in [1,2]:
    # Load up the labview data
    LV = read_csv('../Uniaxial/Uniax_6061_{}/Misc/LV_RAW.dat'.format(x),comment='#',
				index_col=None,header=None,delim_whitespace=True).values
    #[0]Time(s),[1]Crosshead(in),[2]Load(lb),[3]Extensometer,[4]ConcaveAxialSG,[5]ConcaveTransverseSG,[6]ConvexAxialSG,[7]ConvexTransverseSG
    LV = LV[ LV[:,0]>=gotime, :]

    LV[:,1]-=LV[0,1] # Zero crosshead
    LV[:,3]-=LV[0,3] # Zero extensometer
    LV[:,4]*=-1 # Correct that axial SG was reading negative in tension
   
    S = LV[:,2]/1000/area
    eax = n.mean(LV[:,[4,6]],axis=1)
    etr = n.mean(LV[:,[5,7]],axis=1)
    
    # Cut strain gages and stress to below sgcut
    loc = n.nonzero(eax>=sgcut[x])[0][0]
    for e in ['eax','etr','S']:
        exec('{0} = {0}[:loc]'.format(e))

elif x == 3:
    # Load up the results
    LV = read_csv('../Uniaxial/Uniax_6061_{}/Results.dat'.format(x),comment='#',index_col=None,header=None).values
    # [0]Time, [1]NomSts(ksi), [2]Axial Stn (DIC), [3]Transv Stn (DIC), [4]eps_axensometer
    LV = LV[ LV[:,0]>=gotime, :]
    # Truncate at LL
    LV = LV[:LV[:,1].argmax(), :]  
    S = LV[:,1]
    eax = LV[:,2]
    etr = LV[:,3]

# True stress
T = S*(1+eax)
# log strains
eax = n.log(1+eax)
etr = n.log(1+etr)
# plastic strains
eaxp = eax - T/E
etrp = etr - (-ν)*T/E
ezp = -(eaxp+etrp)
# Calculate plastic work in KSI
Wp = cumtrapz(T,eaxp, initial=0)

# [0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, [4]eax_p
# [5]eq_p, [6]ez_p
D = n.c_[Wp, T, eax, etr, eaxp, etrp, ezp]

header = ('[0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, '+
		  '[4]eax_p, [5]eq_p, [6]ez_p')
n.savetxt('../Uniaxial/Uniax_6061_{}/CalData.dat'.format(x), X=D, 
        delimiter=', ', fmt='%.6f',header=header)

# Define our plastic work interpolation
if x ==3 :
    Wp_calcs = n.arange(400,1800,100)/1000
elif x == 2:
    # I know it only got up to 400 psi before SGs popped off
    Wp_calcs = n.arange(200,500,100)/1000
elif x == 1:
    Wp_calcs  = n.arange(400,1100,100)/1000
   
Dint = interp1d(D[:,0],D,axis=0).__call__(Wp_calcs)
# erange1:  depx/depq over whole Wp range
rng = (D[:,0]>=Wp_calcs[0]) & (D[:,0]<=Wp_calcs[-1])
m,b = n.polyfit(eaxp[rng],etrp[rng],1)
erange1 = n.array([m]*Dint.shape[0])
# erange2:  depx/depq over moving window:  prev to next Wp_calcs[k]
erange2 = n.empty_like(erange1)*n.nan
for z in range(1,len(Wp_calcs)-1):
	rng =  (Wp>=Wp_calcs[z-1]) & (Wp<=Wp_calcs[z+1])
	m,b = n.polyfit(eaxp[rng],etrp[rng],1)
	erange2[z] = m

Dint = n.c_[Dint,erange1, erange2]

header = ('[0]Wp (ksi), [1]SigX_Tru (ksi), [2]eax_tot, [3]eq_tot, '+
		  '[4]eax_p, [5]eq_p, [6]ez_p, [7]dexp/deqp, [8]dexp/deqp (moving)')
n.savetxt('../Uniaxial/Uniax_6061_{}/CalData_Iterp.dat'.format(x), X=Dint,
			delimiter=', ', fmt='%.6f',header=header)


if x == 3:  Dint = Dint[::2]

# Plot!
fig, ax1, ax2 = f.make21()

# ax1:  sts-stn with Wp points
ax1.plot(eaxp, T)
cols = []
for k, (e,s,w) in enumerate(zip(Dint[:,4], Dint[:,1], Dint[:,0])):
    l, = ax1.plot(e,s,'o')
    cols.append(l.get_mfc())
    ax1.plot([],[],cols[-1],label='{:.0f}'.format(w*1000))
ax1.set_xlabel('$\\mathsf{e}_\\mathsf{x}^\\mathsf{p}$')
ax1.set_ylabel('$\\sigma_\\mathsf{x}$\n(ksi)')
f.myax(ax1)
f.ezlegend(ax1, title="$\\mathsf{W}_\\mathsf{p} (\\mathsf{psi})$")

# ax2:  stn-stn with Wp points
ax2.plot(eaxp, -etrp)
for k, (e,t,w) in enumerate(zip(Dint[:,4], Dint[:,5], Dint[:,0])):
    ax2.plot(e,-t,'o',mec=cols[k], mfc=cols[k])
ax2.set_xlabel('$\\mathsf{e}_\\mathsf{x}^\\mathsf{p}$')
ax2.set_ylabel('$\\mathsf{e}_\\theta^\\mathsf{p}$')
f.eztext(ax2, '$\\mathsf{de}_\\theta^\\mathsf{p}/\\mathsf{de}_\\mathsf{x}^\\mathsf{p}=.' + 
        '\\mathsf{' + '{:.0f}'.format(-Dint[2,7]*1000) + '}$')
f.myax(ax2)

p.savefig('../Uniaxial/Uniax_6061_{}/CalData.png'.format(x), dpi=100, bbox_inches='tight')
p.close()
