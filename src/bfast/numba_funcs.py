import numpy as np
from numba import njit, prange

@njit
def _Xinv0_numba(x):
    """
    Compute (X^T X)^-1 using pseudo-inverse for stability.
    """
    xtx = x.T @ x
    return np.linalg.pinv(xtx)

@njit
def recresid_numba(x, y, k):
    """
    Numba accelerated recursive residuals.
    Assumes start=k+1 (q=k) and end=n.
    """
    n = x.shape[0]
    # Default q = k (using k points to initialize)
    q = k
    
    if q >= n:
        return np.zeros(0)

    # Pre-allocate result
    rval = np.zeros(n - q)
    
    # Initialize recursion
    y1 = y[:q]
    x_q = x[:q]
    
    # Initial coefficients using Least Squares
    # np.linalg.lstsq returns (coeffs, residuals, rank, s)
    coeffs, _, _, _ = np.linalg.lstsq(x_q, y1)
    
    # Initial inverse covariance
    X1 = _Xinv0_numba(x_q)
    betar = coeffs.copy()
    
    # First residual calculation
    xr = x[q]
    
    # fr = 1 + xr' * X1 * xr
    fr = 1 + xr @ X1 @ xr
    
    # Recursive residual
    rval[0] = (y[q] - xr @ betar) / np.sqrt(fr)
    
    # Loop
    for r in range(q + 1, n):
        # Update X1 using OLD xr and OLD fr
        v_vec = X1 @ xr
        numerator = np.outer(v_vec, v_vec)
        X1 = X1 - numerator / fr
        
        # Update beta using NEW X1 but OLD xr and OLD fr and PREVIOUS residual
        w_vec = X1 @ xr
        update = w_vec * rval[r-q-1] * np.sqrt(fr)
        betar += update
        
        # Now update xr and fr for CURRENT r
        xr = x[r]
        fr = 1 + xr @ X1 @ xr
        val = xr @ betar
        v = (y[r] - val) / np.sqrt(fr)
        rval[r-q] = v
        
    return rval

@njit(parallel=True)
def ssr_triang_numba_loop(n, h, X, y, k):
    """
    Compute full SSR triangular matrix using parallel loop.
    Returns (n, n) matrix where row i contains SSRs for segments starting at i.
    """
    # Initialize with NaNs
    mat = np.full((n, n), np.nan)
    
    # Loop i from 0 to n-h
    # Note: range is exclusive of end, so n-h+1
    for i in prange(n - h + 1):
        # Slice data
        # Note: In parallel loop, creating slices is thread-safe? 
        # Yes, views are safe.
        X_slice = X[i:]
        y_slice = y[i:]
        
        # Call recresid
        # We need k for recresid
        ssr = recresid_numba(X_slice, y_slice, k)
        
        # Fill mat[i]
        # First k elements are NaN (already set)
        # Rest are cumsum(ssr^2)
        # ssr has length L = (n-i) - k
        # We fill mat[i, k : k+L]
        
        # Compute cumsum manually or via np.cumsum (supported in numba)
        ssr_sq = ssr ** 2
        cs = np.cumsum(ssr_sq)
        
        # Assign to matrix row
        # mat[i, k : k + len(cs)] = cs
        # Numba supports slice assignment
        
        # Length check
        if len(cs) > 0:
            mat[i, k : k + len(cs)] = cs
            
    return mat

@njit
def fast_ols(X, y):
    """
    Fast OLS using lstsq.
    Returns coefficients.
    Assumes X and y are clean (no NaNs).
    """
    # np.linalg.lstsq returns (coeffs, residuals, rank, s)
    coeffs, _, _, _ = np.linalg.lstsq(X, y)
    return coeffs

@njit
def fast_predict(X, params):
    """
    Predict y from X and params.
    """
    return X @ params

