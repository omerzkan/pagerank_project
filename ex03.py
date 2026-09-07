
""" ***** PART OF HOMEWORK 3 ***** """

"""
Exercise 3 (page 5):
    "Add a link from page 5 to page 1 in the web of Figure 2.2.  What is the
     dimension of V1(A)?"

Figure 2.2 is

    1 <-> 2        3 <-> 4        5 -> 3,  5 -> 4

and the exercise adds 5 -> 1.  Drawn without arrows the web is now a single
connected piece -- page 5 touches page 1 as well as pages 3 and 4 -- so one
might expect dim V1(A) = 1, a single ranking.

It is still 2.  Nothing links TO page 5, so its row in A is zero
"""

from toolkit import build_A, eig_rank, dim_V1
from webs import EX3_WEB, EX3_N

A = build_A(EX3_WEB, EX3_N)

_, vals = eig_rank(A)
# Only the eigenvalues are used: with dim V1(A) > 1 no single vector is the ranking

print("=== Exercise 3: Figure 2.2 with the link 5 -> 1 added ===")
print("\neigenvalues of A:", vals)
print(f"\ndim V1(A) = {dim_V1(vals)}   -- unchanged, although the web is now connected")
print("\nPage 5 has no backlinks, so row x5 = 0 and the new link carries no weight.\n")