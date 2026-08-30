

import numpy as np


def build_dense_A(links, n):
    
    """ 
    A[i, j] = 1 / n_j if page j links to page i, else 0.
    n_j = number of links on page j.
    """
    A = np.zeros((n, n))
    
    for j, targets in links.items():
        
        if not targets:
            continue
            # This is dangling page j, which has no outgoing links.
        w = 1.0 / len(targets)
        # len(targets) is n_j, the number of links on page j.
        
        for i in targets:
            A[i - 1, j - 1] = w
            
    return A


def eig_rank(A):
    """
    Eigenvector for eigenvalue 1 of A, and scaled to sum 1.
    """
    vals, vecs = np.linalg.eig(A)
    # This returns the eigenvalues and eigenvectors of A. 
    # The eigenvectors are the columns of vecs.

    k = int(np.argmin(np.abs(vals-1.0)))
    # np.argmin selects the minimum element inside the vector, and returns its index.
    # numpy int is different thant he python int, so we convert it to a python int.
    
    v = np.real(vecs[: , k])
    # this get rid of the complex part of the eigenvector to get a real vector
    
    if v.sum() < 0:
        v = -v
    # this ensures that the sum of the vector is positive

    return v / v.sum(), vals 
    # we return the normalized eigenvector and the eigenvalues of A

def build_M(A, m = 0.15):
    """
    M = (1 - m) * A + m * S
    """
    n = A.shape[0]
    return (1 - m) * A + m * np.full((n, n), 1.0/n)
    # here we used np.full to create a matrix of size n x n with all entries equal to 1/n
    # if we had used np.ones((n, n)) instead, we would have had to multiply it by 1/n, which is less efficient.
    # We applied the formula directly
    

def power_method(matvec, n, m=0.15, tol=1e-10, maxit=10_000, x0=None):
    """
    Power method to compute the dominant eigenvector of a matrix.
    matvec: function that computes the matrix-vector product
            In order to be efficient, we do not store the matrix, but we compute the product on the fly.
    n: size of the vector
    m: damping factor
    tol: tolerance for convergence
    maxit: maximum number of iterations, prevents infinite loops
    x0: initial guess for the eigenvector
    """
    
    if x0 is None:
        x = np.full(n, 1.0/n) 
    else:
        x = np.asarray(x0, float)  
    
    x = x / np.sum(x) 
    # Normalize the initial vector as their sum should be 1
 
    history = []

    for k in range(1, maxit+1):
        
        x_new = (1.0-m) * matvec(x) + m/n
        # This is the power method iteration, where we compute the new vector as a combination of the matrix-vector product and the damping factor.
        
        x_new /= np.sum(x_new)  
        # Normalize the new vector
        
        diff = np.abs(x_new - x).sum()
        history.append(diff)
        
        x = x_new
        
        if diff < tol:
            break
        # Stopping criterion: if the difference between the new and old vector is less than the tolerance, we stop iterating.
        
        
    return x, k, diff, history

def build_csr(links, n):
    """
    Build a sparse matrix in Compressed Sparse Row (CSR) format.
    
    we only keep the non-zero entries of the matrix, which are the links between pages.
    In this way, we save memory and computation time, as we do not need to store or compute the zero entries of the matrix.
    
    Return:
        AA (Values) --> the non-zero entries of the matrix
        JA (Column Indices) --> the column indices of the non-zero entries
        IA (Row Pointers) --> the index of the first non-zero entry in each row
    """
    
    counts = np.zeros(n, dtype=np.int64)
    for j, targets in links.items():
        for i in targets:
            counts[i-1] +=1
            # if page j links to page i, we increment the count of non-zero entries in row i-1 (as we are using 0-based indexing)
    
    IA = np.zeros(n+1, dtype=np.int64)
    np.cumsum(counts, out=IA[1:])
    # This is cumulative sum
    nnz = int(IA[n])
    
    JA = np.zeros(nnz, dtype=np.int64)
    AA = np.zeros(nnz, dtype=float)
    # We will fill in the JA and AA arrays with the column indices and values of the non-zero entries of the matrix.
    
    
    pos = IA[: n].copy()
    for j, targets in links.items():
        
        d = len(targets)
        
        if d == 0:
            continue
            # This is a dangling page, which has no outgoing links.
        
        w = 1.0 / d 
        # This is the weight of the link, which is 1/n_j, where n_j is the number of links on page j.
        
        for i in targets:
            # every page i that is linked to by page j, we fill in the corresponding entry in the JA and AA arrays.
            
            p = pos[i - 1]
            # where to put the next non-zero entry in row i-1 (as we are using 0-based indexing)
            
            JA[p] = j-1
            # this non zero entry is in column j-1 (as we are using 0-based indexing)
            
            AA[p] = w
            # this non zero entry has value w
            
            pos[i - 1] = p + 1
            # we increment the position for the next non-zero entry in row i-1
        
    return AA, JA, IA

    
def csr_matvec(AA, JA, IA, x):
    
    """ 
    y[i] = sum over backlinks j of x[j] / n_j, where n_j = number of links on page j.
    
    
    AA --> the non-zero entries of the matrix
    JA --> the column indices of the non-zero entries
    IA --> the index of the first non-zero entry in each row
    x --> the vector to be multiplied by the matrix
    """

    n = len(IA) - 1
    # Finds the number of rows in the matrix, which is the length of IA minus 1, as IA has n+1 entries.
    # It equals the number of pages in the web, which is the same as the length of x.
    
    y = np.zeros(n)
    #  we put our result in y, which is initialized to a zero vector of length n.
    
    for i in range(n):
        s = 0.0
        for p in range(IA[i], IA[i + 1]):
            
            s += AA[p] * x[JA[p]]
            # calculates the score for page i by summing over all the backlinks j of page i, which are stored in the AA and JA arrays.
            
        y[i] = s
    return y

def dangling_mask(links, n):
    """
    Returns a boolean array of length n, where the i-th entry is True if page i is dangling (has no outgoing links), and False otherwise.
    """
    mask = np.zeros(n, dtype=bool)
    
    for j in range(1, n + 1):
        if not links.get(j):
            # if page j has no outgoing links, then it is dangling
            mask[j - 1] = True
            # if page j has no outgoing links, we set the j-1 entry of the mask to True (as we are using 0-based indexing)
    
    return mask


def make_matvec(AA, JA, IA, dmask):
    """
    Returns a function that computes the matrix-vector product for the matrix defined by AA, JA, IA, and the dangling mask.

    mv(x) = A' * x with A' = A + (1/n) * e * d^T, where d^ is the dangling mask and e is the vector of all ones.

    """
    n = len(IA) - 1
    # gives the total number of pages in the web.

    inv_n = 1.0 / n
    # gives the every page equal weight for the dangling pages, which is 1/n.

    def mv(x):
        y = csr_matvec(AA, JA, IA, x)
        # first we compute the matrix-vector product of A and x, which gives us the contribution of the non-dangling pages to the result.
        
        return y + x[dmask].sum() * inv_n
        # adds the contribution of the dangling pages to the result of the matrix-vector product.

    return mv


def load_dat(path):
    
    with open(path, encoding='utf-8', errors="replace") as f:
        
        n, declared = (int(t) for t in f.readline().split())
        # Read the header
        
        urls = {}
        
        for _ in range(n):
            parts = f.readline().split(maxsplit=1)
            # maxsplit=1 ensures that we only split the line into two parts, splitting on the first whitespace
            urls[int(parts[0])] = parts[1].strip() if len(parts) > 1 else ""
        
        links = {k : [] for k in range(1, n + 1)}
        # keys are page numbers, values are the pages each page links TO (outgoing links).
        
        seen, self_loops, dups, read = set(), 0, 0, 0
        
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
                # broken lines we skipped it
                
            s, t = int(p[0]), int(p[1])
            # here s --> source page, t --> target page
            
            read += 1
            # number of read, it should be equal to the declared
            
            if s == t:
                self_loops += 1
                continue
                # self-loop: the paper excludes a link from a page to itself (Sec. 2.1),
                # so the diagonal of A must stay zero.        
            
            if (s, t) in seen:
                dups += 1
                continue
                # duplicate: the model gives each page ONE vote split evenly among its links.
                # Counting the same link twice would give that target a double share.
        
            seen.add((s, t))
            links[s].append(t)
    
    danglings = sum(1 for v in links.values() if not v)
    stats = dict(n=n, declared=declared, read=read, kept=len(seen), self_loops=self_loops, dups=dups, dangling=danglings)
    
    return links, urls, stats