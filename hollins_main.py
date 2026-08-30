

"""
PageRank on the Hollins dataset (6012 pages).
"""
import time
from pathlib import Path

import numpy as np

from toolkit import (load_dat, build_csr, dangling_mask, make_matvec,
                     power_method)

DATA = Path(__file__).parent / "data" / "hollins.dat"   # adjust if needed
DAMPING = 0.15
TOL = 1e-12


""" ************************* Load ************************* """

links, urls, stats = load_dat(DATA)
n = stats["n"]
assert stats["read"] == stats["declared"], "edge count does not match the header"
print("\ndataset :", stats)


""" ************************* Build ************************* """

AA, JA, IA = build_csr(links, n)
dmask = dangling_mask(links, n)
mv = make_matvec(AA, JA, IA, dmask)

csr_mb = (AA.nbytes + JA.nbytes + IA.nbytes) / 1e6
dense_mb = n * n * 8 / 1e6
print(f"nnz     : {len(AA)}   fill = {len(AA) / n**2 * 100:.4f} %")
print(f"storage : CSR {csr_mb:.3f} MB vs dense {dense_mb:.1f} MB "
      f"({dense_mb / csr_mb:.0f}x smaller)")



""" ************************* Solve ************************* """

t0 = time.perf_counter()
x, k, diff, _ = power_method(mv, n, m=DAMPING, tol=TOL)
elapsed = time.perf_counter() - t0

# residual ||Mx - x||_1 : an independent check, NOT the stopping criterion
res = np.abs((1.0 - DAMPING) * mv(x) + DAMPING / n - x).sum()

print(f"\nm = {DAMPING}, tol = {TOL:g}")
print(f"power method  : {k} iterations, {elapsed:.3f} s")
print(f"last increment: {diff:.2e}")
print(f"residual      : {res:.2e}")

assert abs(x.sum() - 1.0) < 1e-10, "scores must sum to 1 (dangling mass leaked?)"
assert x.min() >= DAMPING / n - 1e-12, "every page scores at least m/n [Exercise 9]"
print(f"sum(x) = {x.sum():.12f}   min(x) = {x.min():.3e}   (m/n = {DAMPING / n:.3e})")


""" ************************* Ranking ************************* """
order = np.argsort(-x) 
# argsort returns INDICES and always sorts ascending; the minus sign flips the
# signs so that ascending on -x means descending on x. order[0] = best page.

print(f"\nIf all pages had equal importance, then Uniform score would be 1/n = {1 / n:.6f}\n")
print("Top 10 pages:")

for rank in range(10):
    i = order[rank]           # 0-based index into x
    page = i + 1              # 1-based page number, as in the file
    print(f"{rank + 1}. page {page}   score {x[i]:.6f}   {urls[page]}")
print("\n")
    