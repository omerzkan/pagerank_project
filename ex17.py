
""" ***** PART OF HOMEWORK 1 ***** """

"""
Exercise 17 (p. 10):
    "How should the value of m be chosen?  How does this choice affect the
    rankings and the computation time?"
    
The same web ranked for several values of m.  Small m stays close to the link
structure but converges slowly, because |lambda_2(M)| = (1 - m)|lambda_2(A)|
approaches 1.  Large m converges fast but washes the links out, since M tends
to S and every score tends to 1/n. 

An iteration count means nothing without the tolerance it was measured at, so
TOL is printed with every table.
"""

import time
from pathlib import Path

from toolkit import load_dat, pagerank
from webs import FIG_2_1, FIG_2_1_N

DATA = Path(__file__).parent / "data" / "hollins.dat"
DAMPING_VALUES = [0.05, 0.15, 0.25, 0.50, 0.85, 0.99]
TOL = 1e-10

def study_m(name, links, n, top=5):
    """
    Rank one web for every value of m and print one row per value.
    
    param name: label printed above the table
    param links: dict{page: [pages it links to]}, pages numbered 1 .. n
    param n: number of pages
    param top: how many of the leading pages to print
    """
    print(f"\n--- {name} (n = {n}, tol = {TOL:g}) ---\n")
    print("m \titerations \ttime(s) \ttop pages\n")
    
    for m in DAMPING_VALUES:
        t0 = time.perf_counter()
        x, k, _ = pagerank(links, n, m, TOL)
        print(f"{m:4.2f} \t{k:7d} \t{time.perf_counter() - t0:8.4f} \t{(-x).argsort()[:top] + 1}")

study_m("Figure 2.1", FIG_2_1, FIG_2_1_N, top=3)

links, urls, stats = load_dat(DATA)
study_m("Hollins", links, stats["n"])
print()