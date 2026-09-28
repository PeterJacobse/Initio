import numpy as np
from math import factorial, comb



def Taylor_to_Fourier_matrix(self, N_harmonics: int = 3) -> np.ndarray:
    """
    Constructs the (N+1) x (N+1) matrix A for harmonics n, m in [0, N].
    
    A[n, m] = (2 - delta_{n,0}) / (2^m * Delta! * mu!)
    where Delta = (m - n)/2 and mu = (m + n)/2.
    """
    N = N_harmonics
    A = np.zeros((N + 1, N + 1), dtype = np.float64)
    
    for n in range(N + 1):
        for m in range(n, N + 1):
            if not (m - n) % 2 == 0: continue
            
            delta = (m - n) // 2
            numerator = 1.0 if n == 0 else 2.0
            
            A[n, m] = (numerator / (2 ** m)) * comb(m, delta)
    return A

def Fourier_to_Taylor_matrix_2(self, N_harmonics: int = 3) -> np.ndarray:
    """
    Generates an (N+1) x (N+1) matrix that transforms coefficients 
    from the power basis (x^k) to the Chebyshev basis T_n(x).
    """
    N = N_harmonics
    A = np.zeros((N + 1, N + 1), dtype = np.float64)
    
    for k in range(N + 1):
        power_coeffs = np.zeros(N + 1)
        power_coeffs[k] = 1.0        
        cheb_poly = np.polynomial.Chebyshev.cast(np.polynomial.Polynomial(power_coeffs))        
        c = cheb_poly.coef
        A[:len(c), k] = c
    return A

def Fourier_to_Taylor_matrix(self, N_harmonics: int = 3) -> np.ndarray:
    """
    Generates an (N + 1) x (N + 1) forward Chebyshev coefficient matrix.
    Maps a Chebyshev/Fourier amplitude vector to a power-basis coefficient vector.
    """
    N = N_harmonics
    A = np.zeros((N + 1, N + 1), dtype = np.int64)
    
    for j in range(N + 1):
        t_j = np.polynomial.Chebyshev.basis(j)
        p_j = t_j.convert(kind = np.polynomial.Polynomial)
        coeffs = p_j.coef
        A[:len(coeffs), j] = coeffs
    return A

def numpy_to_latex(self, array: np.ndarray, matrix_type = "bmatrix", precision = 4):
    """
    Converts a 2D numpy array into a LaTeX matrix string.
    
    Parameters:
    - arr: 2D numpy array
    - matrix_type: "bmatrix" [], "pmatrix" (), "vmatrix" | |, or "matrix" (none)
    - precision: Number of decimal places for floats
    """
    if array.ndim != 2: raise ValueError("Array must be 2-dimensional.")
    
    lines = [f"\\begin{{{matrix_type}}}"]
    
    for row in array:
        # Format each element in the row, clean up trailing zeros if they are integers
        row_str = []
        for val in row:
            # Check if it's close to an integer to keep output clean (e.g., 1.0 -> 1)
            if np.isclose(val, round(val)): row_str.append(str(int(round(val))))
            else: row_str.append(f"{val:.{precision}f}".rstrip('0').rstrip('.'))
                
        # Join elements with '&' and end the row with '\\'
        lines.append("  " + " & ".join(row_str) + " \\\\")
        
    lines.append(f"\\end{{{matrix_type}}}")
    return "\n".join(lines)

