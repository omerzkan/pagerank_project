

"""
PageRank on the Hollins dataset (6012 pages).
"""
import time
from pathlib import Path

import numpy as np

from toolkit import load_dat, build_csr, pagerank
                     
DATA = Path(__file__).parent / "data" / "hollins.dat"   # adjust if needed
DAMPING = 0.15
TOL = 1e-12


# =====================================================================
#  LOAD
# =====================================================================
 
links, urls, stats = load_dat(DATA)
n = stats["n"]  # number of pages in the web
print("\ndataset :", stats)

# =====================================================================
#  BUILD -- this is where "Large Scale" shows up
# =====================================================================

AA, JA, IA = build_csr(links, n)

csr_mb = (AA.nbytes + JA.nbytes + IA.nbytes) / 1e6 # total size of our sparse array
dense_mb = n * n * 8 / 1e6 # total size of our dense array

print(f"\nnnz : {len(AA)} fill = {len(AA) / n**2 * 100: .4f}%")
print(f"\nstorage: CSR ({csr_mb:.3f} MB) vs dense ({dense_mb:.1f} MB) -->\t"
      f"({dense_mb / csr_mb:.0f}x smaller)")

# =====================================================================
#  SOLVE
# =====================================================================

# The time covers ranking the web from the link list, so it includes building
# the CSR arrays inside pagerank(), not only the iteration

t0 = time.perf_counter()
x, k, res = pagerank(links, n, m=DAMPING, tol=TOL)
elapsed =time.perf_counter() - t0

print(f"\nm = {DAMPING}, tol = {TOL: g}")
print(f"power method: {k} iterations, {elapsed:.3f} s")
print(f"residual : {res:.2e}  ||Mx-x||_1, not just the step size")

# Every page scores at least m/n, because x = (1-m)Ax + mS with Ax >= 0.
assert x.min() >= DAMPING / n - 1e-12, "every page scores should be at least m/n"
print(f"sum(x) = {x.sum():.12f}  min(x) = {x.min():.3e} (m/n = {DAMPING / n:.3e})")


# =====================================================================
#  RANKING
# =====================================================================
 

order_indexes = np.argsort(-x) 
# argsort returns INDICES and always sorts ascending; the minus sign flips the
# signs so that ascending on -x means descending on x. order[0] = best page.

print(f"\nIf all pages had equal importance, then Uniform score would be 1/n = {1 / n:.6f}\n")
print("Top 10 pages:")

for rank in range(10):
    i = order_indexes[rank]           # 0-based index into x
    page = i + 1              # 1-based page number, as in the file
    print(f"{rank + 1}. page {page} \tscore: \t{x[i]:.6f} \t{urls[page]}")
print("\n")
    