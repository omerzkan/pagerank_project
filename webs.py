"""

webs.py --> every web used in this work

A web is a dict {page: [pages it links to]}, pages numbered 1 .. n as in the paper.
toolkit.build_A and toolkit.build_csr turn them into the matrix


    A[i, j] = 1 / n_j    if page j links to page i. Columns are sources and rows are receivers
    
So column j holds page j's outgoing links and A is column-stochastic
"""

# =====================================================================
# THE TWO WEBS OF THE PAPER
# =====================================================================

# Figure 2.1, page 2 in the paper 
# Four pages: 1 -> 2, 3, 4      2 -> 3, 4      3 -> 1      4 -> 1, 3
# The left page is the source, the right ones are the targets.
FIG_2_1 = {
    1: [2, 3, 4],
    2: [3, 4],
    3: [1],
    4: [1, 3],
}
FIG_2_1_N = 4

# Figure 2.2, page 3  
# Five pages forming TWO disconnected subwebs: W1 = {1, 2}      W2 = {3, 4, 5}
# dim V1(A) = 2 here, so the ranking given by A is not unique.
FIG_2_2 = {
    1: [2],
    2: [1],
    3: [4],
    4: [3],
    5: [3, 4],
}
FIG_2_2_N = 5

# =====================================================================
# THE WEB OF EXERCISES 1, 11 AND 12
# =====================================================================

# Figure 2.1 plus a page 5 that links to page 3, where page 3 also links to page 5:
#     1 -> 2, 3, 4      2 -> 3, 4      3 -> 1, 5      4 -> 1, 3      5 -> 3
# The two new links are 3 -> 5 and 5 -> 3.
MAIN_WEB = {
    1: [2, 3, 4],
    2: [3, 4],
    3: [1, 5],
    4: [1, 3],
    5: [3],
}
MAIN_N = 5

# Exercises 1 and 11 use the SAME web
EX1_WEB,  EX1_N  = MAIN_WEB, MAIN_N
EX11_WEB, EX11_N = MAIN_WEB, MAIN_N

# Exercise 12: the web of Exercise 11 plus a sixth page that 
# links to every other page and to which nobody links.

# CAREFUL, this is a zero ROW, not a zero column: page 6 HAS outgoing links,
# so it is NOT a dangling page, it simply has no backlinks.  
# Its column sums to 1 like every other, but its row is zero

# we used {k: list(v) ...} and we did not use dict(MAIN_WEB): 
# Because a plain dict() copy would share the same list objects with MAIN_WEB,
# so a later edit to one web would silently change the other.
EX12_WEB = {k: list(v) for k, v in MAIN_WEB.items()}
EX12_WEB[6] = [1, 2, 3, 4, 5]
EX12_N = 6

# =====================================================================
# THE WEB OF EXERCISES 2, 3 AND 13
# =====================================================================

# Exercise 2: "Construct a web consisting of three or more subwebs and verify
# that dim(V1(A)) equals (or exceeds) the number of components in the web."
# Three components, each a mutually linking pair:
#     W1 = {1, 2}      W2 = {3, 4}      W3 = {5, 6}

EX2_WEB = {1: [2], 2: [1], 3: [4], 4: [3], 5: [6], 6: [5]}
EX2_N = 6

# Exercise 3: "Add a link from page 5 to page 1 in the web of Figure 2.  The
# resulting web, considered as an undirected graph, is connected.  What is the
# dimension of V1(A)?"
#
# CAREFUL: in Figure 2.2 page 5 already links to pages 3 AND 4.  Adding 5 -> 1
# gives it THREE outgoing links, so its column is (1/3, 0, 1/3, 1/3, 0), not a
# pair of halves.

EX3_WEB = {1: [2], 2: [1], 3: [4], 4: [3], 5: [3, 4, 1]}
EX3_N = 5

# Exercise 13: "Construct a web consisting of two or more subwebs and determine
# the ranking given by formula (3.1)."   Formula (3.1) is M = (1-m)A + mS.
#
# Two components:  W1 = {1, 2}      W2 = {3, 4, 5}
# With A the ranking is not unique (dim V1(A) = 2); with M it is unique and
# positive, and pages in different components become comparable
EX13_WEB = {1: [2], 2: [1], 3: [4], 4: [3], 5: [3]}
EX13_N = 5