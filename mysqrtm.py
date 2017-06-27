def mysqrtm(X):
    from numpy import sqrt, hstack, vstack, dstack
    import numexpr as ne
    ### 0,0
    A = X[:,0,0]
    B = X[:,0,1]
    C = X[:,1,0]
    D = X[:,1,1]
    U00 = ne.evaluate("( (sqrt(A + D - sqrt(A**2 + 4*B*C - 2*A*D + D**2))*(-A + D + sqrt(A**2 + 4*B*C - 2*A*D + D**2)))-((-A + D - sqrt(A**2 + 4*B*C - 2*A*D + D**2))*sqrt(A + D + sqrt(A**2 + 4*B*C - 2*A*D + D**2))) )/(2*sqrt(2)*sqrt(A**2 + 4*B*C - 2*A*D + D**2))")

    ### 0,1
    U01 = ne.evaluate("(B*(-sqrt(A - sqrt(4*B*C + (A - D)**2) + D) + sqrt(A + sqrt(4*B*C + (A - D)**2) + D)))/(sqrt(2)*sqrt(4*B*C + (A - D)**2))")
        
    ### 1,0
    U10 = ne.evaluate("(C*(-sqrt(A - sqrt(4*B*C + (A - D)**2) + D) + sqrt(A + sqrt(4*B*C + (A - D)**2) + D)))/(sqrt(2)*sqrt(4*B*C + (A - D)**2))")
     
    ### 1,1    
    U11 = ne.evaluate("( (A + sqrt(4*B*C + (A - D)**2) - D)*sqrt(A - sqrt(4*B*C + (A - D)**2) + D) + (-A + sqrt(4*B*C + (A - D)**2) + D)*sqrt(A + sqrt(4*B*C + (A - D)**2) + D) )/(2*sqrt(2)*sqrt(4*B*C + (A - D)**2))")

    return hstack((dstack( (U00[:,None][:,:,None],U01[:,None][:,:,None]) ),dstack((U10[:,None][:,:,None],U11[:,None][:,:,None]))))