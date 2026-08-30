import numpy as np

from toolkit import (build_dense_A, eig_rank, build_M, power_method,
                     build_csr, csr_matvec, dangling_mask, make_matvec)
from webs import FIG_2_1, FIG_2_1_N, FIG_2_2, FIG_2_2_N

M_VAL = 0.15
TOL = 1e-12


def dim_V1(vals, eps=1e-9):
    """Number of eigenvalues equal to 1 = dim V1(A)."""
    return int(np.sum(np.abs(vals - 1.0) < eps))
# 1. 'vals - 1.0': Subtract 1.0 from all eigenvalues to check how close they are to 1.
# 3. '< eps': Compare the difference with a tiny threshold (epsilon). We use this because 
#             floating-point rounding errors in computers can make an eigenvalue exactly 
#             equal to 1 appear as 1.0000000001 or 0.9999999999. This allows a safe margin.
# 4. 'np.sum(...)': Sum the True values (where True=1, False=0) to count how many eigenvalues equal 1.
# 5. 'int(...)': Cast the final sum into a standard integer and return it.

""" ************************* Figure 2.1 ************************* """
 
print("=== Figure 2.1 ===")
A = build_dense_A(FIG_2_1, FIG_2_1_N)

""" (1) eigenvector of A for lambda = 1 is [12, 4, 9, 6]/31  [p. 3] """
xA, vals = eig_rank(A)
assert dim_V1(vals) == 1, "Fig 2.1 should have dim V1(A) = 1"
assert np.allclose(xA, np.array([12, 4, 9, 6]) / 31), "A-ranking is wrong"
print("A ranking  :", np.round(xA, 6), "  = [12,4,9,6]/31")

"""# (2) M matches the matrix printed in Example 1 [p. 6]"""
PAPER_M = np.array([
    [0.0375,  0.0375, 0.8875, 0.4625],
    [0.32083, 0.0375, 0.0375, 0.0375],
    [0.32083, 0.4625, 0.0375, 0.4625],
    [0.32083, 0.4625, 0.0375, 0.0375],
])
M = build_M(A, M_VAL)
assert np.allclose(M, PAPER_M, atol=1e-5), "M does not match the paper"
print("M matches the paper, max entrywise diff =", f"{np.abs(M - PAPER_M).max():.2e}")

""" (3) power method -> 0.368, 0.142, 0.288, 0.202  [p. 6] """
x, k, diff, _ = power_method(lambda v: A @ v, FIG_2_1_N, m=M_VAL, tol=TOL)
assert np.allclose(x, [0.368, 0.142, 0.288, 0.202], atol=1e-3), "M-ranking is wrong"
# 1. 'np.allclose(...)': Checks if all elements in two arrays are element-wise equal within a tolerance.
# We use this instead of '==' because floating-point operations can introduce 
# tiny rounding errors (e.g., 0.36800000001 instead of exactly 0.368).

# 2. 'atol=1e-3': Sets the Absolute Tolerance to 0.001 (10^-3). It means a difference of up to 
# 0.001 between the calculated value and the expected value is acceptable.

assert abs(x.sum() - 1.0) < 1e-12, "scores must sum to 1"
print(f"M ranking  : {np.round(x, 6)}   ({k} iterations)")

""" (4) the CSR path must reproduce the dense path """
AA, JA, IA = build_csr(FIG_2_1, FIG_2_1_N)
dmask = dangling_mask(FIG_2_1, FIG_2_1_N)
assert not dmask.any(), "Fig 2.1 has no dangling nodes"

v = np.random.default_rng(0).random(FIG_2_1_N)
assert np.allclose(A @ v, csr_matvec(AA, JA, IA, v)), "CSR product != dense product"

x_csr, k_csr, _, _ = power_method(make_matvec(AA, JA, IA, dmask),FIG_2_1_N, m=M_VAL, tol=TOL)
assert np.allclose(x, x_csr, atol=1e-10), "CSR path gives a different ranking"
print(f"CSR path reproduces the dense path ({k_csr} iterations)")


""" ************************* Figure 2.2 ************************* """

print("\n=== Figure 2.2 ===")
A2 = build_dense_A(FIG_2_2, FIG_2_2_N)

""" (5) two disconnected subwebs -> dim V1(A) = 2  [p. 4] """
_, vals2 = eig_rank(A2)
assert dim_V1(vals2) == 2, "Fig 2.2 should have dim V1(A) = 2"
print("dim V1(A) =", dim_V1(vals2), " -> the ranking given by A is NOT unique")

""" (6) with M the ranking is unique: 0.2, 0.2, 0.285, 0.285, 0.03   [p. 6] """

x2, k2, _, _ = power_method(lambda v: A2 @ v, FIG_2_2_N, m=M_VAL, tol=TOL)
assert np.allclose(x2, [0.2, 0.2, 0.285, 0.285, 0.03], atol=1e-8), "M-ranking is wrong"
print(f"M ranking  : {np.round(x2, 6)}   ({k2} iterations)")


print("\nAll checks passed.")