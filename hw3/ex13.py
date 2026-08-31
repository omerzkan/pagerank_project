import numpy as np

from webs import FIG_2_2, FIG_2_2_N
from toolkit import build_dense_A, power_method


# --------------------------------------------------
# FIGURE 2.2 WEB
# --------------------------------------------------

links = FIG_2_2
n = FIG_2_2_N

m = 0.15


# --------------------------------------------------
# BUILD LINK MATRIX A
# --------------------------------------------------

A = build_dense_A(links, n)

print("Link matrix A:")
print(A)


# --------------------------------------------------
# POWER METHOD
# --------------------------------------------------

x, iterations, error, history = power_method(
    lambda v: A @ v,
    n,
    m=m,
    tol=1e-10,
    maxit=1000
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nPageRank vector:")
print(x)

print("\nIterations:")
print(iterations)

print("\nPage scores:")

for i, score in enumerate(x, start=1):
    print(f"Page {i}: {score:.6f}")


# --------------------------------------------------
# RANKING
# --------------------------------------------------

ranking = np.argsort(-x)

print("\nRanking:")

for position, page in enumerate(ranking, start=1):
    print(
        f"{position}. Page {page + 1} "
        f"(score = {x[page]:.6f})"
    )