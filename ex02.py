
""" ***** PART OF HOMEWORK 3 ***** """

"""
Exercise 2 (page 5):
    "Construct a web consisting of three or more subwebs and verify that
     dim(V1(A)) equals (or exceeds) the number of the disconnected subwebs."

Our web is three separate pairs, each pair linking to itself only:

    1 <-> 2        3 <-> 4        5 <-> 6

No link crosses from one pair to another, so A is block diagonal with three
2x2 blocks.  Each block is column-stochastic on its own, so each one carries
its own eigenvector for lambda = 1, zero outside its block.  Those three
vectors are independent, so dim V1(A) = 3.
"""

from toolkit import build_A, eig_rank, dim_V1
from webs import EX2_WEB, EX2_N

SUBWEBS = 3 # Our subwebs  1 <-> 2        3 <-> 4        5 <-> 6

A = build_A(EX2_WEB, EX2_N)

_, vals = eig_rank(A)
# Only the eigenvalues are used here. Since dim V1(A) > 1, the vector eig_rank
# picks out of the eigenspace is one of infinitely many: it is not a ranking

print("\n=== Exercise 2: a web made of three subwebs ===")
print("\neigenvalues of A:", vals)
print(f"\ndim V1(A) = {dim_V1(vals)}      number of subwebs = {SUBWEBS}")

assert dim_V1(vals) >= SUBWEBS, "dim V1(A) is smaller than the number of subwebs"
print("\nVerified: dim V1(A) equals (or exceeds) the number of subwebs.\n")