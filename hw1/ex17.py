"""
Exercise 17 (Bryan & Leise, p. 10)

"How should the value of m be chosen? How does this choice affect the rankings
and the computation time?"

We run the same solver on three webs for several values of m and report the
number of iterations, the time, and the top pages.

IMPORTANT: an iteration count is meaningless without the tolerance it was
measured at, so TOL is printed with every table.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time

import numpy as np
from pathlib import Path

from toolkit import (load_dat, build_csr, dangling_mask, make_matvec, power_method)

from webs import FIG_2_1, FIG_2_1_N, FIG_2_2, FIG_2_2_N

ROOT = Path(__file__).resolve().parent.parent      # Code_HW1_HW3/
DATA = ROOT / "data" / "hollins.dat"

M_VALUES = [0.15, 0.25, 0.50, 0.85, 0.99]
TOL = 1e-10

def study(name, links, n, top=3):
    """Run the power method for every m and print one row per value."""
    AA, JA, IA = build_csr(links, n)
    dmask = dangling_mask(links, n)
    mv = make_matvec(AA, JA, IA, dmask)

    print(f"\n--- {name}   (n = {n}, tol = {TOL:g}) ---")
    print("   m   iterations    time (s)   top pages")

    for m in M_VALUES:
        t0 = time.perf_counter()
        x, k, _, _ = power_method(mv, n, m=m, tol=TOL)
        elapsed = time.perf_counter() - t0

        order = np.argsort(-x)
        best = [int(order[r]) + 1 for r in range(top)]
        print(f"{m:5.2f}   {k:8d}    {elapsed:8.4f}   {best}")
    print("\n")
        

""" ******************* The two webs from the paper ******************* """

study("Figure 2.1", FIG_2_1, FIG_2_1_N)
study("Figure 2.2", FIG_2_2, FIG_2_2_N)


""" ******************* The Hollins dataset ******************* """

links, urls, stats = load_dat(DATA)
study("Hollins", links, stats["n"], top=5)