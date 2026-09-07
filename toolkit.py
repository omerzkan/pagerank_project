

import numpy as np

# =====================================================================
# 1.  DENSE PART -- small webs only
# =====================================================================

def build_A(links, n):
    
    """
    Build the link matrix A of equation 2.1

    A[i, j] = 1 / n_j if page j links to page i, 0 otherwise
    
    where n_j is the number of outgoing links of page j
    
    A column of zeros means page j is dangling.
    
    param links: dict {page: [pages it links to]}, pages numbered 1, .. , n
    param n: number of pages
    return: A: 2D-array of shape (n, n)
    """
    A = np.zeros((n, n)) 
    # we initialize the matrix A with zeros, then we will fill
    for j, targets in links.items():
        
        if not targets:
            continue
            # this is dangling page, there is no outgoing links, empty column
        
        w = 1.0 / len(targets)
        
        for i in targets:
            A[i-1, j-1] = w
        
    return A
    
def build_M(A, m=0.15):
    
    """
    Build M = (1-m)A + m S, equation 3.1 of paper where every entry of S equal to 1/n.
    
    Used only to print M and compare it with the matrices of Example 1 page 6 and equation 3.3
    
    The solver never builds M: for n=6012, because it would be really heavy for computer memory

    param A: link matrix, 2D-array of shape (n, n)
    param m: damping factor, 0.15 in the paper
    return M; 2D-array of shape (n, n)
    """
    
    n = A.shape[0]
    
    return (1.0 - m) * A + m * np.full((n, n), 1.0/n)

def eig_rank(A):
    """
    This is numpy's eigensolver, not our algorithm.
    It is used the way np.linalg.det is used next to a hand-written determinant: to confirm.
    
    This is for CROSS-CHECKING the results of our power method, not for the actual computation of PageRank.
    
    param A: link matrix, 2D-array of shape (n, n)
    return x: eigenvector for the eigenvalue closest to 1, scaled to sum 1
    return vals: all n eigenvalues of A
    """
    
    vals, vecs = np.linalg.eig(A)
    # calculate all eigenvalues and eigenvectors of A
    
    k = int(np.argmin(np.abs(vals - 1.0)))
    # find the index of the eigenvalue closest to 1
    
    v = np.real(vecs[:, k])
    # take the real part of the eigenvector corresponding to the eigenvalue closest to 1
    
    if v.sum() < 0:
        v = -v
        # if the sum of the eigenvector is negative, we flip its sign to make it positive
    
    x = v / v.sum()
    # scale the eigenvector to sum to 1
    
    return x, vals


def dim_V1(vals, tol=1e-9):
    """
    dimV1(A) = how many eigenvalues of A are equal to 1
    
    In floating point an eigenvalue is never exactly 1,
    so equal to 1 has to be read as |lambda - 1| < tol.
    
    A tolerance is what turns a numerical quantity into a structural one here: 
    dim V1(A) = 1 means the ranking is unique, dim V1(A) > 1 means it is not unique.
    
    param vals: eigenvalues, as returned by eig_rank
    param tol: an eigenvalue counts as 1 when |lambda - 1| < tol
    return dim: dimension of the eigenspace of A for the eigenvalue 1
    """
    
    dimension_v1 = int(np.sum(np.abs(vals-1.0) < tol))
    return dimension_v1

# =====================================================================
# 2.  SPARSE PART -- CSR built by hand, no scipy
# =====================================================================


def build_csr(links, n):
    
    """
    Store A in compressed sparse row format, as three arrays AA, JA, IA.
    
        AA[k] --> the k-th non-zero value of A, read row by row
        JA[k] --> the column index of that value
        IA[i] --> IA[i]: i. satirin AA dizisindeki başlangic indeksini tutar.
                i. satirdaki elemanlar AA[IA[i] : IA[i+1]] dilimindedir (slice).
                Dizinin son elemani (IA[n]) toplam sifir disi eleman sayisina (nnz) eşittir.
        
    row i of A holds the backlinks of page i, column j of A holds the outgoing links of page j.
    
    so we first count how many backlinks each page has, then fill the arrays in a second pass
    
    Hollins data: 6012 x 6012 = 36 million entries, only 23875 of them non-zero
    
    param links: dict {page: [pages it links to]}, pages numbered 1, .., n
    param n: number of pages
    return AA: 1D-array of non-zero values of A
    return JA: 1D-array of their column indices
    return IA: 1D-array of pointers, length n+1
    """
    
    # ----- first pass: count backlinks of each page, to fill IA
    counts = np.zeros(n, dtype=np.int64)
    for j, targets in links.items():
        for i in targets:
            counts[i-1] += 1
            # for each target page i, we increment its backlink count (0-based indexing)

    # ----- second pass: fill IA

    IA = np.zeros(n+1, dtype=np.int64)
    np.cumsum(counts, out=IA[1: ])
    # IA[i] will hold the starting index in AA for row i, and IA[n] will be the total number of non-zero entries (nnz)
    # out = IA[1:] means we are storing the cumulative sum starting from the second element of IA, leaving IA[0] as 0
    nnz = int(IA[n])
    
    AA = np.zeros(nnz, dtype=float)
    JA = np.zeros(nnz, dtype=np.int64)
    
    # ---- third pass: fill AA and JA
    next_index = IA[:n].copy()
    # next_index[i] will point to the next available position in AA and JA for row i
    # copying IA[:n] ensures we don't modify IA while filling AA and JA
    
    for j, targets in links.items():
        if not targets:
            continue
            # if page j has no outgoing links, we skip it because it is dangling and its column full zero
        
        w = 1.0 / len(targets)
        for i in targets:
            p = next_index[i-1] # targets 1 based and next_index is 0 based, so we subtract 1
            AA[p] = w # fill the value in AA
            JA[p] = j - 1 # fill the column index in JA (0-based indexing)
            next_index[i-1] = p + 1 
    
    return AA, JA, IA

def csr_matvec(AA, JA, IA, x):
    """
    Compute y = A x with A stored in CSR. 
    row i only touches only its own non-zero entries
    
    y[i] = sum over k = IA[i] ... IA[i+1]-1 of AA[k] * x[JA[k]]
    
    param AA: 1D-array non-zero values of A
    param JA: their column indices
    param IA: row pointers, length n+1
    param x: 1D-array of length n
    return y: 1D-array of length n, equal to A x
    """
    
    n = len(IA) - 1
    y = np.zeros(n)

    for i in range(n):
        start, end = IA[i], IA[i+1]
        y[i] = AA[start : end] @ x[JA[start: end]]
        # the @ operator performs matrix multiplication, 
        # which in this case is a dot product between the non-zero values of row i and the corresponding entries in x.
    
    return y

def dangling_pages(links, n):
    """
    Find the pages with no outgoing links, the zero columns of A.
    Hollins has 3189 of them
    
    param links: dict {page: [pages it links to]}, pages numbered 1, .., n
    param n: number of pages
    return dangling: 1D-array of the 0-based indices of the dangling pages
    """
    return np.array([j - 1 for j in range(1, n + 1) if not links.get(j)], dtype=np.int64)
    
# =====================================================================
# 3.  SOLVER -- the power method on equation (3.2)
# =====================================================================

def pagerank(links, n, m=0.15, tol=1e-10, maxit=10000, x0=None):
    
    """
    Compute the PageRank vector x by iterating equation (3.2) until convergence.
    
    x_{k+1} = (1-m) A x_k + m S    S = (1/n, 1/n, ..., 1/n)^T
    
    The matrix M = (1-m) A + m S is never built. This is the only solver in the project
    every web, small, or large, is ranked by this function.
    
    DANGLING NODES: A page with no outgoing links gives a zero column, 
    so A is only substochastic and every iteration loses the mass sitting on those pages.
    
    Line b gives that mass back, spread uniformly over all pages.
    This is the same as replacing A by A + (1/n) e d^T, 
    where d is the indicator vector of dangling pages and e is the vector of ones.
    The corrected matrix is column-stochastic, and the power method converges to a unique solution.
    
    WHAT res MEASURES
    
    Line c makes x_{k+1} exactly M x k, so 
    res = ||x_{k+1} - x_k ||_1 = ||M x k - x_k ||_1
    
    So res is not merely "the iterate stopped moving": it is the residual of the eigenvalue problem
    M x = x, measured at x_k. Since the vector we return is x_{k+1}, one step further on, its own residual
    is smaller still -- res is a conservative bound on the error of the answer, not just of the step.
     
    param links: dict {page: [pages it links to]}, pages numbered 1, .. , n
    param n: number of pages
    param m: damping factor, 0.15 in the paper
    param tol: stop when the residual is below this tolerance res < tol
    param maxit: stop after this many iterations, even if not converged (prevents infinite loops)
    param x0: initial guess, 1D-array of length n, if None 1/n
    
    return x: 1D-array of length n, the PageRank vector, scaled to sum 1 (x>=0 and sum(x)=1)
    return k: number of iterations performed
    return res: the residual of the last iterate, ||M x_k - x_k||_1,
    if the residual is still large the iteration DID NOT CONVERGE
    """
    
    AA, JA, IA = build_csr(links, n)
    dangling = dangling_pages(links, n)
    
    # "x= x0 / sum" rather than "x0 /= sum" --> the caller's vector is not touched
    x = np.full(n, 1.0/n) if x0 is None else np.asarray(x0, dtype=float)
    x = x/x.sum()
    
    for k in range(1, maxit + 1):
        
        y = csr_matvec(AA, JA, IA, x)   # STEP A: A x
        y = y + x[dangling].sum() / n   # STEP B: Give back the dangling mass
        x_new = (1.0 - m) * y + m/n     # STEP C: Equation 3.2
        
        x_new /= x_new.sum()
        
        # The sum is already 1 in exact arithmetic, thanks to STEP B
        # This only removes rounding drift
        
        res = np.linalg.norm(x_new - x, 1) # 1-norm definition 4.1 of the paper
        
        x = x_new
        
        if res < tol:
            break
    return x, k, res
        
        


# =====================================================================
# 4.  READING THE DATASET
# =====================================================================

def load_dat(path):
    
    """
    Read the .dat file (Hollins.dat) in the assignment
    
    hollins.dat
        line 1 --> n, the number of pages, and the declared link count
        next n lines --> page index and its URL
        rest --> one link per line, "source target", 1-based
    
    Two kinds of line are dropped which are stated in the paper:
        -Self-loops --> does not count a link from a page to itself
                        (you don't vote for yourself, you vote for others)
        -Duplicates --> a page casts one vote split evenly among its links,
                        so counting the same link twice would give that target a double share. 
    param path: path to the .dat file
    return links: dict {page: [pages it links to]}, pages numbered 1, .. , n
    return urls: dict {page: url}
    return stats: dict with n, declared, read, kept, self_loops, dups, dangling     
    """
    
    with open(path, encoding="utf-8", errors="replace") as f:
        
        n, declared = (int(t) for t in f.readline().split())
        
        urls = {}
        for _ in range(n):
            parts = f.readline().split(maxsplit=1)
            urls[int(parts[0])] = parts[1].strip() if len(parts) > 1 else ""
        
        links = {k : [] for k in range(1, n+1)}
        seen, self_loops, dups, read= set(), 0, 0, 0
        
        for line in f:
            p = line.split()
            if len(p) != 2:
                continue # skip broken lines
            
            s, t = int(p[0]), int(p[1]) # s = source page, t = target page
            read += 1
            
            if s==t:
                self_loops += 1
                continue # skip self-loops
            
            if (s, t) in seen:
                dups += 1
                continue # skip duplicates
            
            seen.add((s, t))
            links[s].append(t)
        
    stats = {
        "n": n,
        "declared": declared,
        "read": read,
        "kept": len(seen),
        "self_loops": self_loops,
        "dups": dups,
        "dangling": len(dangling_pages(links, n))
    }
        
    return links, urls, stats