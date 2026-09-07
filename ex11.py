
""" ***** PART OF HOMEWORK 1 ***** """

"""
Exercise 11 (page 8):
    "Consider again the web in Figure 2.1, with the addition of a page 5 that
     links to page 3, where page 3 also links to page 5.  Calculate the new
     ranking by finding the eigenvector of M (corresponding to lambda = 1)
     that has positive components summing to one.  Use m = 0.15."

Same web as Exercise 1; what changes is the matrix.  Exercise 1 ranked it with
A alone, here with M = (1 - m) A + m S.  Our power method iterates equation
(3.2), x <- (1 - m) A x + m s, so M itself is never built.
"""

import numpy as np
from toolkit import pagerank
from webs import EX11_WEB, EX11_N

DAMPING = 0.15
TOL = 1e-12

x, k, res = pagerank(EX11_WEB, EX11_N, m=DAMPING, tol=TOL)

print(f"\n=== Exercise 11: the same web ranked with M, m = {DAMPING} ===")
print(f"\n{k} iterations, residual {res:.1e}\n")

for page in range(1, EX11_N + 1):
    print(f"  page {page}:  {x[page - 1]:.6f}")
    
print(f"\nsum = {x.sum():.6f} smallest component = {x.min():.6f} > 0")
order_indices = np.argsort(-x)
print("\nranking, best first: ", [int(i)+1 for i in order_indices], "\n")
# the printed values the numbers of the pages (Which are sorted best to worst)