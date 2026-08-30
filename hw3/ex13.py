
"""
!!! EXERCISE 13 ANLAMADIM !!!
"""

"""
Exercise 13 -- one ranking on disconnected subwebs
HW3   |   Zeynep

  "Construct a web consisting of two or more subwebs and determine the
   ranking given by formula (3.1)."

Five pages, W1 = {1,2}, W2 = {3,4,5}.
ANSWER: x = (1/5, 1/5, 54/185, 1029/3700, 3/100),  ranking 3 > 4 > 1 = 2 > 5.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import sympy as sp
from toolkit import *
from webs import *

m = sp.Rational(15, 100)
A = build_A(EX13_WEB, EX13_N)
print("web:", EX13_WEB)
print("A ="); print(sp.pretty(A))
print()
print("dim V1(A) =", eigenspace_dim(A), " <- disconnected: (2.1) gives NO ranking")
print("dim V1(M) =", eigenspace_dim(build_M(A, m)), " <- (3.1) gives exactly one")
print()
M = np.array(build_M(A, m).tolist(), dtype=float)
print("M = 0.85 A + 0.15 S,  m/n = 0.03")
for row in M:
    print("   " + "  ".join("%5.2f" % v for v in row))
print()
x = rank_with_M(A, m)
q, k, res = power_method(A, m=0.15, tol=1e-14)
print("exact  :", [sp.nsimplify(v) for v in x])
print("decimal:", [round(float(v), 6) for v in x])
print("power  :", np.round(q, 6), "  %d iterations, residual %.2e" % (k, res))
print()
order = np.argsort(-q) + 1
print("ranking:", " > ".join("page %d" % p for p in order))
print()
print("x5 =", sp.nsimplify(x[4]), "= m/n =", sp.nsimplify(m / EX13_N),
      " <- page 5 has no backlinks (Exercise 9)")
print("x1 = x2 = 1/5 exactly, because the subweb {1,2} is symmetric.")
