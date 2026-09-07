"""
Exercise 12 (page 8):
    "Add a sixth page that links to every page in the previous exercise (Exercise 11), but to
     which no other page links.  Rank the pages using A, then using M with
     m = 0.15, and compare the results."

Page 6 HAS outgoing links, so it is not a dangling node.  What it has is an
empty ROW: nobody links to it.  A dangling page is an empty COLUMN.  The two
faults are opposites, and the code treats them differently: an empty column is
repaired inside pagerank(), an empty row is left alone and produces exactly the
score this exercise is about.

Both rankings come from the same solver.  With m = 0, equation (3.2) reads
x <- A x, which is formula (2.1); with m = 0.15 it is formula (3.2) itself.
"""
import numpy as np
from toolkit import pagerank
from webs import EX12_WEB, EX12_N

DAMPING = 0.15
TOL = 1e-12

xA, kA, _ = pagerank(EX12_WEB, EX12_N, m=0.0, tol=TOL)
xM, kM, _ = pagerank(EX12_WEB, EX12_N, m=DAMPING, tol=TOL)

print("\n=== Exercise 12: a sixth page that nobody links to ===")
print(f"\nwith A: {kA} iterations, m = {0.0} \twith M: {kM} iterations, m = {DAMPING}\n")

print("page \twith A \t\twith M")
for page in range(1, EX12_N + 1):
    print(f"{page} \t{xA[page - 1]:.6f} \t{xM[page - 1]:.6f}")

print(f"\npage 6 with A: {xA[5]:.6f}    no backlinks, so its score is 0")
print(f"\npage 6 with M: {xM[5]:.6f}    exactly m/n = {DAMPING / EX12_N:.6f}")

print("\norder with A:", np.argsort(-xA) + 1)
print("\norder with M:", np.argsort(-xM) + 1)
print("\nSame order: damping lifts page 6 off zero without reordering the rest.\n")