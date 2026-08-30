
"""
Exercise 12 (Bryan & Leise, p. 8)

"Add a sixth page that links to every page in the previous exercise, but to
which no other page links. Rank the pages using A, then using M with m = 0.15,
and compare the results."

Note: page 6 HAS outgoing links, so it is not a dangling node. What it has is
an empty row: nobody links to it.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from toolkit import build_dense_A, eig_rank, power_method
from webs import EX11_WEB, EX11_N, EX12_WEB, EX12_N

DAMPING = 0.15
TOL = 1e-12

""" ******************* Rank the 6-page web with A and with M ******************* """

A6 = build_dense_A(EX12_WEB, EX12_N)

xA, _ = eig_rank(A6) # formula (2.1)
xA[np.abs(xA) < 1e-12] = 0.0

xM, k, _, _ = power_method(lambda v: A6 @ v, EX12_N, m=DAMPING, tol=TOL)   # formula (3.2)

print(f"\nm = {DAMPING},  M-ranking converged in {k} iterations\n")
print(" page       with A       with M")
for page in range(1, EX12_N + 1):
    print(f"   {page}      {xA[page - 1]:.6f}     {xM[page - 1]:.6f}")

""" ******************* What happened to page 6 ******************* """

print(f"\npage 6 with A: {xA[5]:.6f}   -> no backlinks, so its score is 0")
print(f"page 6 with M: {xM[5]:.6f}   -> exactly m/n = {DAMPING / EX12_N:.6f}  [Exercise 9]")


""" ******************* What happened to pages 1-5 ******************* """

# rank the 5-page web of Exercise 11 again, to compare
A5 = build_dense_A(EX11_WEB, EX11_N)
x5, _, _, _ = power_method(lambda v: A5 @ v, EX11_N, m=DAMPING, tol=TOL)

ratio = xM[:5] / x5
print("\nExercise 12 score / Exercise 11 score, for pages 1-5:")
print(np.round(ratio, 10))
print(f"all equal, and the common value is 1 - m/n = {1 - DAMPING / EX12_N}")
print("=> adding page 6 rescales the old ranking; it does not reorder it.\n")