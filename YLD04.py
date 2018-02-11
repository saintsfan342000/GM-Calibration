import sympy as sp

'''
This file symbolically computes the yield function PHI and the flow-rule ratios, then saves the result as a string to Yld04.txt.  That can then be read into a file with the same symbols defined.

The point here is so that I don't have to have all these symbolic definitions in my ErrorMin file.
'''

a = 8

sr, sx, sq = sp.var('sr, sx, sq')
(cp12,cp13,cp21,cp23,cp31,
        cp32,cp44,cp55,cp66) = sp.var("cp12,cp13,cp21,cp23,cp31,cp32,cp44,cp55,cp66")
(cpp12,cpp13,cpp21,cpp23,cpp31,
        cpp32,cpp44,cpp55,cpp66) = sp.var("cpp12,cpp13,cpp21,cpp23,cpp31,cpp32,cpp44,cpp55,cpp66")
cp44, cp55, cpp44, cpp55, cp66, cpp66 = 1, 1, 1, 1, 1, 1

Cp = sp.zeros(6,6)
Cp[0,1], Cp[0,2] = -cp12, -cp13
Cp[1,0], Cp[1,2] = -cp21, -cp23
Cp[2,0], Cp[2,1] = -cp31, -cp32
Cp[3,3], Cp[4,4], Cp[5,5] = cp44, cp55, cp66

Cpp = sp.zeros(6,6)
Cpp[0,1], Cpp[0,2] = -cpp12, -cpp13
Cpp[1,0], Cpp[1,2] = -cpp21, -cpp23
Cpp[2,0], Cpp[2,1] = -cpp31, -cpp32
Cpp[3,3], Cpp[4,4], Cpp[5,5] = cpp44, cpp55, cpp66

T = sp.zeros(6,6)
T[0,0], T[0,1], T[0,2] = 2, -1, -1
T[1,0], T[1,1], T[1,2] = -1, 2, -1
T[2,0], T[2,1], T[2,2] = -1, -1, 2
T[3,3], T[4,4], T[5,5] = 3, 3, 3
T*=sp.Rational(1,3)

s = sp.Matrix([sr, sq, sx, 0, 0, 0])
Sp = (Cp*T*s)[:3]
Spp = (Cpp*T*s)[:3]

PHI = ( (Sp[0]-Spp[0])**a + 
      (Sp[0]-Spp[1])**a + 
      (Sp[0]-Spp[2])**a + 
      (Sp[1]-Spp[0])**a + 
      (Sp[1]-Spp[1])**a + 
      (Sp[1]-Spp[2])**a + 
      (Sp[2]-Spp[0])**a + 
      (Sp[2]-Spp[1])**a + 
      (Sp[2]-Spp[2])**a
    )

dPHI = PHI.diff(sq)/PHI.diff(sx)

# Write PHI followed by PHI,sq / PHI,sx
with open('Yld04.txt', 'w') as fid:
    fid.write(str(PHI))
    fid.write('\n')
    fid.write(str(dPHI))
    fid.close()

