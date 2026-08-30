"""
A web is a dict {page: [pages it links to]}.

# A[i][j] = 1/n_j if page j links to page i. Columns are sources.
# Column-stochastic, following Bryan & Leise eq. (2.2).
"""

# ---------- from the paper ----------

# Figure 2.1, page 2. Four pages.
#   1 -> 2, 3, 4      2 -> 3, 4      3 -> 1      4 -> 1, 3
# For example: 1 --> 2, 3, 4 means 1 is source and 2, 3, 4 are targets.
# Gives exactly the matrix A of equation (2.2).
FIG_2_1 = {
    1: [2, 3, 4],
    2: [3, 4],
    3: [1],
    4: [1, 3],
}
FIG_2_1_N = 4

# Figure 2.2, page 3. Five pages, two disconnected subwebs.
#   W1 = {1, 2}   W2 = {3, 4, 5}
FIG_2_2 = {
    1: [2],
    2: [1],
    3: [4],
    4: [3],
    5: [3, 4],
}
FIG_2_2_N = 5

# Figure 2.1 plus a page 5 that links to page 3, where page 3 also links to page 5.
#  1 -> 2, 3, 4      2 -> 3, 4      3 -> 1, 5      4 -> 1, 3      5 -> 3
# 3 --> 5 and 5 --> 3 are new
MAIN_WEB = {
    1: [2, 3, 4],
    2: [3, 4],
    3: [1, 5],        # <- was [1]
    4: [1, 3],
    5: [3],
}
MAIN_N = 5


""" EXERCISE 11 Web """
EX11_WEB, EX11_N = MAIN_WEB, MAIN_N        # Ex 11: rank it with M
EX1_WEB, EX1_N = MAIN_WEB, MAIN_N          # Ex 1 : rank it with A

""" EXERCISE 12 Webs """

# Exercise 12: the web of Exercise 11 plus a sixth page that links to every
# other page and to which nobody links. A zero ROW, not a zero column.
# It is not a dangling page, there is no backling for page 6.
EX12_WEB = dict(MAIN_WEB)
EX12_WEB[6] = [1, 2, 3, 4, 5]
EX12_N = 6

""" EXERCISE 3 Web """

# Exercise 3: the web of Figure 2.2 with a link added from page 5 to page 1.
# CAREFUL: in Figure 2.2 page 5 links to pages 3 AND 4. Adding 5 -> 1 gives it
# THREE outgoing links, so its column is (1/3, 0, 1/3, 1/3, 0), not a pair of halves.
EX3_WEB = {1: [2], 2: [1], 3: [4], 4: [3], 5: [3, 4, 1]}
EX3_N = 5



"""
!!! EXERCISE 2 WEBI ANLAMADIM !!!
"""
""" EXERCISE 2 Web """
# Exercise 2: three disconnected subwebs, each a mutually linking pair.
#   W1 = {1,2}   W2 = {3,4}   W3 = {5,6}      ->  dim V1(A) = 3
EX2_WEB = {1: [2], 2: [1], 3: [4], 4: [3], 5: [6], 6: [5]}
EX2_N = 6

"""
!!! EXERCISE 13 WEBI ANLAMADIM !!!
"""
""" EXERCISE 13 Web """
# Exercise 13: two disconnected subwebs, ranked with M.
#   W1 = {1,2}   W2 = {3,4,5}
# Page 5 has no backlinks, so it scores exactly m/n = 3/100.
EX13_WEB = {1: [2], 2: [1], 3: [4], 4: [3], 5: [3]}
EX13_N = 5
