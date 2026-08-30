"""
Exercise 3 -- connectivity is not enough
HW3   |   Zeynep

  "Add a link from page 5 to page 1 in the web of Figure 2.2. The resulting
   web, considered as an undirected graph, is connected. What is the
   dimension of V1(A)?"

CAREFUL: in Figure 2.2 page 5 links to pages 3 AND 4. Adding 5 -> 1 gives it
THREE outgoing links, so column 5 is (1/3, 0, 1/3, 1/3, 0) -- not a pair of
halves. ANSWER: dim V1(A) = 2.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sympy as sp
from toolkit import *
from webs import *

print("Figure 2.2 :", FIG_2_2, "   <- page 5 links to 3 AND 4")
print("with 5 -> 1:", EX3_WEB, "   <- page 5 now has THREE outgoing links")
print()
A = build_A(EX3_WEB, EX3_N)
print("A ="); print(sp.pretty(A))
print("column 5 =", [sp.nsimplify(A[i, 4]) for i in range(EX3_N)])
print()
print("dim V1(A) =", eigenspace_dim(A))
print("basis:")
for b in (A - sp.eye(EX3_N)).nullspace():
    print("   ", [sp.nsimplify(v) for v in b])
print()
print("Row 5 of A is zero, so x5 = 0: page 5 has outgoing links but no")
print("backlinks. It is transient -- it feeds both {1,2} and {3,4} without")
print("ever connecting them. Undirected connectivity is not the right")
print("condition; two separate absorbing sets remain, hence dim = 2.")
