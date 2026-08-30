"""
Exercise 1 (Bryan & Leise, p. 5)

"Suppose the people who own page 3 in the web of Figure 1 are infuriated by the
fact that its importance score, computed using formula (2.1), is lower than the
score of page 1. In an attempt to boost page 3's score, they create a page 5
that links to page 3; page 3 also links to page 5. Does this boost page 3's
score above that of page 1?"

Formula (2.1) means the plain link matrix A, without the damping factor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from toolkit import build_dense_A, eig_rank
from webs import FIG_2_1, FIG_2_1_N, EX1_WEB, EX1_N


""" ******************* Original web (Figure 2.1) ******************* """

A_old = build_dense_A(FIG_2_1, FIG_2_1_N)
x_old, _ = eig_rank(A_old)

print("\nOriginal web, 4 pages:")
for page in range(1, FIG_2_1_N + 1):
    print(f"  page {page}: {x_old[page - 1]:.6f}")


""" ******************* Web with the new page 5 ******************* """

A_new = build_dense_A(EX1_WEB, EX1_N)
x_new, _ = eig_rank(A_new)

print("\nWeb with the new page 5, 5 pages:")
for page in range(1, EX1_N + 1):
    print(f"  page {page}: {x_new[page - 1]:.6f}")

# the exact answer is a vector of fractions with denominator 49
print("\nsame scores multiplied by 49:", np.round(x_new * 49, 6))


""" ******************* Answer ******************* """

x1 = x_new[0]
x3 = x_new[2]

print(f"\nx3 = {x3:.6f}   vs   x1 = {x1:.6f}")
if x3 > x1:
    print("Answer: YES, the attack works. Page 3 now outranks page 1.\n")
else:
    print("Answer: NO, page 1 still outranks page 3.\n")