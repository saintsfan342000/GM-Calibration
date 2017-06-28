import numpy as n
pi = n.pi

def Iterate(P, F, to, R, ex, eq, E, v):
    '''
    Iterative scheme to calculate:
        - True thickness
        - True stresses
        - Log plastic strains
    Requires:
        - Pressure P
        - Force F
        - *Initial* thickness, to
        - Current radius, R
        - Axial log stn, ex
        - Hoop log stn, eq
        - Modulus
        - Poisson's ratio
    Returns:
        - t_tru, tau_x, tau_q, ep_x, ep_q, ep_3, e3
        
        '''
    # This first iteration was pulled out of while so that
    # there's no need for a if k==0 check in the loop
    # Initialize to erroneously high value
    t_tru = 10000 
    #Initial approximation of thickness
    ta = to*n.exp(-ex-eq)
    # Approximate true stress based on ta
    tau_x = P*R/(2*ta) + F/(2*pi*R*ta)
    tau_q = P*R/ta
    # e_plastic strain is e_tot minus e_elastic
    ep_x = ex - (tau_x - v*tau_q)/E 
    ep_q = eq - (tau_q - v*tau_x)/E 
    # Assume plastic incompressibility
    ep_3 = -(ep_x + ep_q) 
    # Then add on the elastic part of e3 (plane stress) to get e3_total
    e3 = ep_3 - (v/E)*(tau_x + tau_q)   
    t_tru = to*n.exp(e3)  #Get new thickness
     
    itcount = 1
    itmax = 200
    #print('ta: ', ta)
    #print('t_tru:  ', t_tru)
    while (min(ta/t_tru, t_tru/ta)<.999) and (itcount<=itmax):
        ta = t_tru
        # Approximate true stress based on ta
        tau_x = P*R/(2*ta) + F/(2*pi*R*ta) 
        tau_q = P*R/ta
        # e_plastic strain is e_tot minus e_elastic
        ep_x = ex - (tau_x - v*tau_q)/E 
        ep_q = eq - (tau_q - v*tau_x)/E 
        # Assume plastic incompressibility
        ep_3 = -(ep_x + ep_q) 
        # Then add on the elastic part of e3 (plane stress) to get e3_total
        e3 = ep_3 - (v/E)*(tau_x + tau_q)   
        t_tru = to*n.exp(e3)  #Get new thickness
        itcount+=1

    if (min(ta/t_tru, t_tru/ta)<.999):
        print('Max iteration reached')
        
    return t_tru, tau_x, tau_q, ep_x, ep_q, ep_3, e3

