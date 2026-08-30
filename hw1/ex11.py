"""
Exercise 11 (Bryan & Leise, p. 8)

"Consider again the web in Figure 2.1, with the addition of a page 5 that links
to page 3, where page 3 also links to page 5. Calculate the new ranking by
finding the eigenvector of M (corresponding to lambda = 1) that has positive
components summing to one. Use m = 0.15."
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from toolkit import build_dense_A, build_M, eig_rank, power_method
from webs import EX11_WEB, EX11_N

M_VAL = 0.15
TOL = 1e-12


""" ******************* Ranking with M ******************* """

A = build_dense_A(EX11_WEB, EX11_N)

# power_method applies x <- (1-m) A x + m s, i.e. equation (3.2),
# so it never forms M explicitly.
x, k, diff, _ = power_method(lambda v: A @ v, EX11_N, m=M_VAL, tol=TOL)

print(f"\nm = {M_VAL}, tol = {TOL:g}, converged in {k} iterations")
for page in range(1, EX11_N + 1):
    print(f"  page {page}: {x[page - 1]:.6f}")

print(f"\nsum = {x.sum():.6f}      min = {x.min():.6f}  (all components positive)")


""" ******************* Cross-check with a direct eigensolver ******************* """

M = build_M(A, M_VAL)
x_eig, _ = eig_rank(M)
print(f"max difference vs. numpy.linalg.eig: {np.abs(x - x_eig).max():.2e}")


""" ******************* Ranking ******************* """

order = np.argsort(-x)
print("ranking, best first:", [int(i) + 1 for i in order])
print("\n")