"""
This is the verification of the code on the two webs of the paper
"""

import numpy as np
 
from toolkit import build_A, build_M, eig_rank, dim_V1, build_csr, csr_matvec, pagerank
from webs import FIG_2_1, FIG_2_1_N, FIG_2_2, FIG_2_2_N
 
DAMPING = 0.15
TOL = 1e-12 # stopping tolerance of the power method

def check(claim, computed, expected, atol):
    
    """
    Compare a computed quantity with the value the paper prints, print the
    largest difference, and stop the script if it exceeds atol.
 
    HOW atol IS CHOSEN.  A value the paper prints with d decimals carries a
    rounding error of at most 0.5 * 10^-d, so even a perfect computation can
    differ from it by that much.  We therefore allow 10^-d, one unit in the
    last printed decimal, which leaves a factor-2 margin:  1e-3 for 0.368,
    1e-5 for 0.32083.  Where the paper's value is exact -- a fraction like
    12/31, or 0.455 = 0.85 * 0.5 + 0.03 -- we allow 1e-10, well above our
    own rounding noise (~1e-16) and the power method's stopping tolerance.
    The paper marks the distinction itself: it writes "x1 ~ 0.368" for the
    rounded values and "x1 = 0.2" for the exact ones.
 
    The tolerance is ABSOLUTE (the 'a' in atol) because rounding to d decimals
    is an absolute error, the same for every entry whatever its size; a
    relative tolerance would be unfair to the small entries.
 
    :param claim: what is verified, printed on the report line
    :param computed: our value, a numpy array
    :param expected: the paper's value
    :param atol: largest difference allowed, chosen as explained above
    """
    diff = np.abs(computed - expected).max()
    assert diff < atol, f"{claim}: off by {diff}, allowed {atol}"
    print(f"{claim: <48s} max diff {diff:.1e}")
    

# =====================================================================
#  FIGURE 2.1 -- four pages, no dangling nodes, unique ranking
# =====================================================================
 
print("=== Figure 2.1 ===")
 
A = build_A(FIG_2_1, FIG_2_1_N)
 
# The eigenvector of A for lambda = 1 is [12, 4, 9, 6]/31, p. 3.  Exact.
xA, vals = eig_rank(A)
check("eigenvector of A = [12,4,9,6]/31, page 3", xA, np.array([12, 4, 9, 6]) / 31, 1e-10)
assert dim_V1(vals) == 1
print("dim V1(A) =", dim_V1(vals), " -> the ranking given by A is unique")

# M as printed in Example 1, page 6 -- five decimals, so 1e-5.  This one check this.
PAPER_M = np.array([
    [0.0375,  0.0375, 0.8875, 0.4625],
    [0.32083, 0.0375, 0.0375, 0.0375],
    [0.32083, 0.4625, 0.0375, 0.4625],
    [0.32083, 0.4625, 0.0375, 0.0375],
])
check("M matches Example 1, page 6", build_M(A, DAMPING), PAPER_M, 1e-5)

# The ranking, page 6.
x, k, res = pagerank(FIG_2_1, FIG_2_1_N, m=DAMPING, tol=TOL)
check("PageRank ~ 0.368, 0.142, 0.288, 0.202, p. 6", x, [0.368, 0.142, 0.288, 0.202], 1e-3)
print("computed", np.round(x, 6), "iterations", k, "residual", f"{res:.1e}")

# build_csr and csr_matvec are written by hand and
# independently of build_A, so their product must agree with A @ v on any vector
AA, JA, IA = build_csr(FIG_2_1, FIG_2_1_N)
np.random.seed(0)
v = np.random.rand(FIG_2_1_N)
check("hand-written CSR product equals A @ v", csr_matvec(AA, JA, IA, v), A @ v, 1e-10)

# =====================================================================
#  FIGURE 2.2 -- two disconnected subwebs, so A does NOT give a ranking
# =====================================================================
 
print("\n=== Figure 2.2 ===")
 
A2 = build_A(FIG_2_2, FIG_2_2_N)
 
# The point of this web: V1(A) is two-dimensional, so the ranking given by A is
# not unique, page 4.
_, vals2 = eig_rank(A2)
assert dim_V1(vals2) == 2
print("dim V1(A) =", dim_V1(vals2), " -> the ranking given by A is NOT unique")
 
# M as printed in equation (3.3), page 6.
PAPER_M2 = np.array([
    [0.03, 0.88, 0.03, 0.03, 0.03 ],
    [0.88, 0.03, 0.03, 0.03, 0.03 ],
    [0.03, 0.03, 0.03, 0.88, 0.455],
    [0.03, 0.03, 0.88, 0.03, 0.455],
    [0.03, 0.03, 0.03, 0.03, 0.03 ],
])
check("M matches equation 3.3, page 6", build_M(A2, DAMPING), PAPER_M2, 1e-10)
 
# With M the ranking exists and is unique, page 6 -- "x1 = 0.2", exact.
x2, k2, res2 = pagerank(FIG_2_2, FIG_2_2_N, m=DAMPING, tol=TOL)
check("PageRank = 0.2, 0.2, 0.285, 0.285, 0.03, p. 6", x2, [0.2, 0.2, 0.285, 0.285, 0.03], 1e-10)
print("computed", np.round(x2, 6), "iterations", k2, "residual", f"{res2:.1e}")
 
# =====================================================================
#  A WEB WITH A DANGLING NODE -- not in the paper, but Hollins is 53% dangling
# =====================================================================

# Figure 2.1 with the link 3 -> 1 removed: page 3 now has no outgoing links, so
# column 3 of A is zero.  Neither web of the paper has a dangling node, so
# without this block the dangling correction inside pagerank() would be
# untested -- it could be deleted and every check above would still pass.
DANGLING_WEB, DANGLING_N = {1: [2, 3, 4], 2: [3, 4], 3: [], 4: [1, 3]}, 4

print("\n=== Dangling node ===")

A_fixed = build_A(DANGLING_WEB, DANGLING_N).copy()
A_fixed[:, 2] += 1.0 / DANGLING_N
check("column sums after the correction are 1", A_fixed.sum(axis=0), 1.0, 1e-10)

x_dense, _ = eig_rank(build_M(A_fixed, DAMPING))
xd, kd, resd = pagerank(DANGLING_WEB, DANGLING_N, m=DAMPING, tol=TOL)
check("pagerank matches the dense corrected matrix", xd, x_dense, 1e-10)
print("computed", np.round(xd, 6), "iterations", kd, "residual", f"{resd:.1e}")

print("\nAll checks passed.")