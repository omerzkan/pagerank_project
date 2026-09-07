"""
Exercise 13 (page 8):
    "Construct a web consisting of two or more subwebs and determine the
     ranking given by formula (3.1)."

Our web is two subwebs with no link between them:

    1 <-> 2            3 <-> 4  <-  5

Formula (2.1) cannot rank it: dim V1(A) = 2, so any mixture of the two subwebs'
own eigenvectors is an equally valid answer.  Formula (3.1), M = (1 - m)A + mS,
has every entry positive, so dim V1(M) = 1 by Lemma 3.2 and the ranking below
is the only one.
"""

from toolkit import build_A, eig_rank, dim_V1, pagerank
from webs import EX13_WEB, EX13_N

DAMPING = 0.15
TOL = 1e-12

_, vals = eig_rank(build_A(EX13_WEB, EX13_N))
x, k, res = pagerank(EX13_WEB, EX13_N, m=DAMPING, tol=TOL)

print("\n=== Exercise 13: a web made of two subwebs ===")
print(f"\ndim V1(A) = {dim_V1(vals)} --> formula (2.1) gives no unique ranking")

print(f"\nranking with formula (3.1), m = {DAMPING}: {k} iterations, residual {res:.1e}\n")
for page in range(1, EX13_N + 1):
    print(f"page {page}:  {x[page - 1]:.6f}")
    
print("\npages 1 and 2 tie exactly: the subweb 1 <-> 2 is symmetric")
print(f"\npage 5 has no backlinks, so its score is exactly m/n = {DAMPING / EX13_N:.2f}\n")