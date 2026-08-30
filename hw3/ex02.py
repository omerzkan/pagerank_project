
"""
!!! EXERCISE 2 ANLAMADIM !!!
"""

"""
Exercise 2 -- disconnected subwebs inflate the eigenspace
HW3   |   Zeynep

  "Construct a web consisting of three or more subwebs and verify that
   dim(V1(A)) equals (or exceeds) the number of the components."

Six pages, three disconnected pairs.   ANSWER: dim V1(A) = 3.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sympy as sp
from toolkit import *
from webs import *

A = build_A(EX2_WEB, EX2_N)
print("web:", EX2_WEB)
print("A ="); print(sp.pretty(A))
print("column sums:", [sum(A[:, j]) for j in range(EX2_N)])
print()
print("dim V1(A) =", eigenspace_dim(A), "  <- three components, three dimensions")
print()
print("basis of V1(A) (each vector lives in one block):")
for b in (A - sp.eye(EX2_N)).nullspace():
    print("   ", [sp.nsimplify(v) for v in b])
print()
print("rank_with_A refuses, and that is the point of the exercise:")
try:
    rank_with_A(A)
except ValueError as e:
    print("   ", e)
