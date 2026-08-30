PageRank -- CLA4LSP
Ozkan, Ulas

Implementation of the PageRank algorithm described in
K. Bryan and T. Leise, "The $25,000,000,000 Eigenvector: The Linear Algebra
Behind Google", SIAM Review 48 (2006), 569-581.

Python 3 with NumPy only. No sparse-matrix library is used: the CSR storage
and the sparse matrix-vector product are implemented by hand in toolkit.py.


FILES
-----
toolkit.py        All shared code:
                    build_dense_A   link matrix A,          eq. (2.1), (2.2)
                    build_M         M = (1-m)A + mS,        eq. (3.1)
                    power_method    iterates eq. (3.2); M is never formed
                    build_csr       CSR arrays AA, JA, IA, built by hand
                    csr_matvec      sparse product A*x
                    dangling_mask   pages with no outgoing links
                    make_matvec     A*x plus the dangling-node correction
                    load_dat        reader for the .dat dataset format
                    eig_rank        direct eigensolver, used only to cross-check

webs.py           Every web used in this work, stored as link lists
                  {page: [pages it links to]}: the two webs of the paper
                  (Figures 2.1 and 2.2) and the webs built for the exercises.

test_paper.py     Verification. Every assertion is a value printed in the
                  paper: the eigenvector [12,4,9,6]/31, the matrix M of
                  Example 1, the rankings of Examples 1 and 2, and
                  dim V1(A) = 2 for Figure 2.2. It also checks that the
                  hand-written CSR path reproduces the dense path.

hollins_main.py   PageRank on the supplied dataset (6012 pages): storage
                  comparison, iteration count, timing, residual, and the
                  ten highest-ranked pages.

hw1/              Mandatory Homework HW1
    ex11.py         Exercise 11
    ex17.py         Exercise 17

hw3/              Optional Homework HW3, we choose solving 10 additional exercise from PageRank Paper
    ex01.py         Exercise 1
    ex02.py         Exercise 2
    ex03.py         Exercise 3
    ex12.py         Exercise 12
    ex13.py         Exercise 13

data/
    hollins.dat   The dataset supplied with the assignment.


HOW TO RUN
----------
From this folder, with no arguments:

    python test_paper.py        run this first, it validates the code
    python hollins_main.py
    python hw1/ex11.py
    python hw3/ex01.py          ... and so on

Each file is standalone and prints its own results.


NOTES
-----
- Iteration counts depend on the tolerance. test_paper.py, hollins_main.py
  and the exercise files use tol = 1e-12; ex17.py uses tol = 1e-10 because
  it runs the solver many times. Every table prints the tolerance it used.

- The paper assumes a web with no dangling nodes. The supplied dataset has
  3189 of them out of 6012, so make_matvec applies the identity

      A'x  =  A x  +  (1/n) e (d^T x)

  which is exactly the effect of replacing every zero column of A by 1/n,
  without ever forming the dense matrix.