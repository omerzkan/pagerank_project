"""
Exercise 1 (page 5):
    "Suppose the people who own page 3 in the web of Figure 2.1 are infuriated
     by the fact that its importance score, computed using formula (2.1), is
     lower than the score of page 1.  In an attempt to boost page 3's score,
     they create a page 5 that links to page 3; page 3 also links to page 5.
     Does this boost page 3's score above that of page 1?"
"""

from toolkit import build_A, eig_rank, dim_V1, pagerank
from webs import FIG_2_1, FIG_2_1_N, EX1_WEB, EX1_N

TOL = 1e-12

def rank_with_A(links, n, title):
    """
    Rank a web with formula (2.1) and print the scores page by page
    
    param links: dict {page: [pages it links to]}, pages number 1 .. n
    param n: number of pages
    param title: label printed above the scores
    return x: the ranking, x >= 0 and sum(x) = 1
    """
    x, k, res = pagerank(links, n, m=0.0, tol=TOL)
    print(f"\n{title} -- {n} pages, {k} iterations, residual {res:.1e}")
    for page in range(1, n+1):
        print(f"page {page}: {x[page-1]:.6f}")
        
    return x


print("=== Exercise 1: does the extra page 5 boost page3? ===")

x_old = rank_with_A(FIG_2_1, FIG_2_1_N, "BEFORE, web of FIGURE 2.1")
x_new = rank_with_A(EX1_WEB, EX1_N, "AFTER, page 5 added")

print(f"\nBefore:\nPage 3 = {x_old[2]:.6f}\nPage 1 = {x_old[0]:.6f}")
print(f"\nAfter:\nPage 3 = {x_new[2]:.6f}\nPage 1 = {x_new[0]:.6f}")

print("\nAnswer:", "YES, page 3 now outranks page " if x_new[2] > x_old[2] else "NO, page 1 still outranks page 3\n")
