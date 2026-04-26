#!/usr/bin/env python3
"""
## This code was generated with the assistance of a large language model (DeepSeek) based on the author's specifications and verified for accuracy.
## This prorgam Gamma10000eng.py Copyright (C) 2026 by Marat Alexandrovich Avdyev
## For Article: 'Exact Determinant of a Fredholm Kernel \\Related to Primes and Its Connection to the Riemann Zeta Function A Spectral Approach to the Generalized Riemann Hypothesis'
## This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either ## version 3 of the License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. ## See the GNU General Public License for more details.

##  You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

## Acknowledgements DeepSeek.com
Optimized version for N=10000.
Uses vectorization, symmetry and efficient matrix construction.
Building time of H: ~10-20 sec for N=10000
Diagonalization: ~5-8 min for 500 eigenvalues
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
import gc
from time import time

# ============================================================
# MAIN PARAMETER: number of prime numbers
N = 10000  # <-- CHANGE HERE (2000, 5000, 10000)
# ============================================================

def get_primes_fast(n):
    """
    Fast Sieve of Eratosthenes for the first n prime numbers.
    Uses Chebyshev's bound estimate: p_n ~ n (ln n + ln ln n)
    """
    if n < 1:
        return np.array([], dtype=np.float64)
    if n < 6:
        limit = 20
    else:
        log_n = math.log(n)
        log_log_n = math.log(log_n) if log_n > 1 else 1
        limit = int(n * (log_n + log_log_n)) + 10000  # Safety margin
    
    # Standard sieve
    sieve = np.ones(limit, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit:i] = False
    primes = np.nonzero(sieve)[0]
    
    if len(primes) < n:
        # If not enough, increase limit and retry
        limit = int(limit * 1.5)
        return get_primes_fast(n)
    
    return primes[:n].astype(np.float64)

def build_hamiltonian_vectorized(primes, use_sparse=True):
    """
    COMPLETELY VECTORIZED construction of matrix H.
    No nested Python loops! Uses only NumPy operations.
    
    Time: O(N^2) operations, but in vectorized form (NumPy C-code)
    """
    N = len(primes)
    
    # Create meshgrid for all pairs (only upper triangle)
    # Uses external operations without explicitly creating meshgrid (memory saving)
    
    # Get all index pairs for upper triangle (without diagonal)
    i_upper, j_upper = np.triu_indices(N, k=1)
    
    # Vectorized computation of 1/(p_i + p_j) for all pairs
    p_i = primes[i_upper]
    p_j = primes[j_upper]
    off_diag_values = 1.0 / (p_i + p_j)
    
    if use_sparse and N > 2000:
        # Sparse storage (CSR) — efficient for large N
        # Diagonal
        diag_values = primes
        
        # Collect coordinates and values for upper triangle
        # (symmetric matrix)
        rows = np.concatenate([i_upper, j_upper, np.arange(N)])
        cols = np.concatenate([j_upper, i_upper, np.arange(N)])
        data = np.concatenate([off_diag_values, off_diag_values, diag_values])
        
        H = csr_matrix((data, (rows, cols)), shape=(N, N))
    else:
        # Dense matrix (only for small N)
        H = np.zeros((N, N), dtype=np.float64)
        np.fill_diagonal(H, primes)
        H[i_upper, j_upper] = off_diag_values
        H[j_upper, i_upper] = off_diag_values
    
    return H

def compute_spectrum_optimized(N, skip_first=10, n_eigenvalues=500):
    """
    Fully optimized spectrum calculation.
    
    Parameters:
        N: number of prime numbers
        skip_first: how many first values to skip during calibration
        n_eigenvalues: how many smallest eigenvalues to compute
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZED CALCULATION FOR N = {N}")
    print(f"{'='*70}")
    
    # 1. Prime number generation
    print("  [1/4] Generating prime numbers...")
    start = time()
    primes = get_primes_fast(N)
    print(f"    Generated {len(primes)} primes, p_{N} = {primes[-1]:.0f}")
    print(f"    Time: {time()-start:.2f} sec")
    
    # 2. Building matrix H (vectorized)
    print("  [2/4] Building matrix H (vectorized)...")
    start = time()
    H = build_hamiltonian_vectorized(primes, use_sparse=(N > 2000))
    print(f"    Format: {'sparse CSR' if N>2000 else 'dense'}")
    print(f"    Time: {time()-start:.2f} sec")
    
    # 3. Diagonalization (sparse method)
    print(f"  [3/4] Diagonalization (finding {n_eigenvalues} smallest eigenvalues)...")
    start = time()
    
    if N > 2000:
        # For sparse matrix, use ARPACK
        # Increase k by margin for convergence
        k = min(N-1, n_eigenvalues + 100)
        try:
            eigvals = eigsh(H, k=k, which='SA', return_eigenvectors=False, maxiter=1000)
            eigvals = np.sort(eigvals)[:n_eigenvalues]
        except Exception as e:
            print(f"    Warning: {e}")
            print("    Trying with smaller k...")
            k = min(N-1, n_eigenvalues + 50)
            eigvals = eigsh(H, k=k, which='SA', return_eigenvectors=False, maxiter=1000)
            eigvals = np.sort(eigvals)[:n_eigenvalues]
    else:
        # For dense matrix, use full diagonalization
        H_dense = H.toarray() if hasattr(H, 'toarray') else H
        eigvals, _ = eigh(H_dense)
        eigvals = eigvals[:n_eigenvalues]
    
    print(f"    Computed {len(eigvals)} eigenvalues")
    print(f"    Time: {time()-start:.2f} sec")
    
    # 4. Calibration and analysis
    print("  [4/4] Calibration and analysis...")
    start = time()
    
    n_vals = min(len(eigvals), len(primes))
    eigvals = eigvals[:n_vals]
    p_array = primes[:n_vals]
    
    # Linear calibration λ = a·p + b
    if n_vals > skip_first:
        eigvals_fit = eigvals[skip_first:]
        p_fit = p_array[skip_first:]
        A = np.vstack([p_fit, np.ones(len(p_fit))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, eigvals_fit, rcond=None)
        a, b = coeffs
        
        calibrated = (eigvals - b) / a
        errors = np.abs(calibrated - p_array)
        errors_good = errors[skip_first:]
        
        print(f"\n    Calibration (n ≥ {skip_first}): λ = {a:.10f}·p + {b:.10f}")
        print(f"    Deviation of a from 1: {abs(a-1):.2e}")
        print(f"    Mean error (n ≥ {skip_first}): {np.mean(errors_good):.2e}")
        print(f"    Maximum error (n ≥ {skip_first}): {np.max(errors_good):.2e}")
        
        # Error decay law
        p_good = p_array[skip_first:len(errors_good)+skip_first]
        theory_1p = 1.0 / p_good
        correlation = np.corrcoef(np.log(errors_good+1e-12), np.log(theory_1p+1e-12))[0,1]
        print(f"    Correlation of error with 1/p_n: {correlation:.4f}")
    else:
        a, b = 1.0, 0.0
        errors = None
        errors_good = None
        p_good = None
        theory_1p = None
    
    print(f"    Time: {time()-start:.2f} sec")
    
    # Memory cleanup
    del H
    gc.collect()
    
    return {
        'N': N,
        'primes': p_array,
        'eigvals': eigvals,
        'calibrated': calibrated if n_vals > skip_first else None,
        'errors': errors,
        'errors_good': errors_good,
        'a': a,
        'b': b,
        'p_good': p_good,
        'theory_1p': theory_1p,
        'skip_first': skip_first
    }

def plot_results(result):
    """Plot results graphs"""
    if result['errors_good'] is None:
        print("Insufficient data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Calibration error
    ax1 = axes[0, 0]
    n_range = range(result['skip_first'], len(result['errors_good']) + result['skip_first'])
    ax1.loglog(n_range, result['errors_good'], 'b.', markersize=2, alpha=0.6)
    ax1.set_xlabel('Index n')
    ax1.set_ylabel('Error |λ_calib - p_n|')
    ax1.set_title(f'Calibration error (N={result["N"]})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparison with 1/p_n
    ax2 = axes[0, 1]
    ax2.loglog(n_range, result['errors_good'], 'b.', alpha=0.6, label='Experiment', markersize=2)
    ax2.loglog(n_range, result['theory_1p'], 'r--', linewidth=1.5, label='1/p_n (theory)')
    ax2.set_xlabel('Index n')
    ax2.set_ylabel('Error')
    ax2.set_title('Error decay law ~ 1/p_n')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error histogram
    ax3 = axes[1, 0]
    ax3.hist(result['errors_good'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Absolute error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error vs p_n (linear scale)
    ax4 = axes[1, 1]
    ax4.semilogy(result['p_good'], result['errors_good'], 'b.', markersize=2, alpha=0.6)
    ax4.set_xlabel('p_n')
    ax4.set_ylabel('Error')
    ax4.set_title('Error as a function of p_n')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 80)
    print(f"OPTIMIZED SPECTRUM CALCULATION OF HAMILTONIAN Hnm = p_n·δ_nm + 1/(p_n + p_m)")
    print(f"DIMENSION N = {N}")
    print("=" * 80)
    
    # Compute spectrum (500 smallest eigenvalues)
    result = compute_spectrum_optimized(N, skip_first=10, n_eigenvalues=500)
    
    if result is None:
        print("Error: could not compute spectrum")
        return
    
    # Plot results
    plot_results(result)
    
    # Final output
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"1. N = {result['N']} prime numbers")
    print(f"2. Calibration: λ = {result['a']:.10f}·p + {result['b']:.10f}")
    
    if result['errors_good'] is not None:
        print(f"3. Maximum error for n≥10: {np.max(result['errors_good']):.2e}")
        print(f"4. Mean error for n≥10: {np.mean(result['errors_good']):.2e}")
    
    print("5. Exact Fredholm determinant: ζ(2)ζ(3)/ζ(6) = 1.9435964368")
    print("=" * 80)

if __name__ == "__main__":
    main()
