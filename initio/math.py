import numpy as np
from math import comb
from sympy import Expr, Symbol
import matplotlib.colors as mcolors



def Lorentzian(x: float | Symbol | np.ndarray, center: float, gamma: float) -> float | Expr | np.ndarray:
    """A normalized Lorentzian function centered around x0 with HWHM gamma.

    Args:
        x (float | sympy.Symbol): Variable. Passing a float will evaluate the Lorentzian at that point and return a float. Passing a np.ndarray yields a np.ndarray. Passing a sympy.Symbol will generate a sympy.Expr object
        center (float): Center of the Lorentzian function
        gamma (float): Width (half-width at half-maximum) of the Lorentian function

    Returns:
        float | Expr
    """
    gamma2 = gamma ** 2
    return (1.0 / (np.pi * gamma)) * (gamma2 / ((x - center) ** 2 + gamma2)) # type: ignore

def Lorentzian_Taylor_coefficients(center: float, gamma: float, x0: float, N: int) -> np.ndarray:
    """Obtain the Taylor coefficients for a normalized Lorentzian function of width gamma, centered at V_center, for a Taylor expansion around V0

    Args:
        center (float): Center of the Lorentzian function
        gamma (float): Width (half-width at half-maximum) of the Lorentian function
        x0 (float): Taylor expansion center
        N (int): Number of coefficients

    Returns:
        np.ndarray: Vector of Taylor coefficients
    """
    d = (center - x0) - 1j * gamma
    n = np.arange(N)
    return np.imag((-1.0) ** n / d ** (n + 1)) / np.pi

def Lorentzian_Fourier_amplitudes(center: float = 0, gamma: float = .05, x0: float = 0., dx: float = 0., N_harmonics: int = 32) -> np.ndarray:
    """Obtain the amplitudes of the harmonics of an integrated Lorentzian transfer function subjected to a harmonic perturbation

    Args:
        center (float, optional): Center of the Lorentzian function. Defaults to 0.
        gamma (float, optional): Width (half-width at half-maximum) of the Lorentian function. Defaults to .05.
        x0 (float, optional): Center value of the harmonic modulation. Defaults to 0..
        dx (float, optional): Amplitude of the harmonic modulation. Defaults to 0..
        N_harmonics (int, optional): Number of harmonics. Defaults to 32.

    Returns:
        np.ndarray: Vector of Fourier amplitudes
    """
    V_p_tilde = center - 1j * gamma # Complex peak voltage
    DeltaV_tilde = V_p_tilde - x0 # Complex difference between peak voltage and dc bias

    s = np.sqrt(DeltaV_tilde ** 2 - dx ** 2) # Square root of difference of squares    
    s = np.where(s.imag > 0, -s, s) # Pick the negative root

    xi = DeltaV_tilde + s # Angle factor
    phi = -np.angle(xi)
    
    n = np.arange(1, N_harmonics)
    amplitudes = (dx ** (n - 1) / np.abs(xi) ** n) * np.sin(n * phi) / n    
    return 2 * amplitudes / np.pi

def diagonalQ(matrix: np.ndarray) -> bool:
    """
    Returns False if matrix has off-diagonal entries, and True otherwise.

    Args:
        matrix (np.ndarray): input matrix

    Returns:
        bool
    """
    return not bool(np.any(matrix - np.diag(np.diag(matrix))))

def Taylor_to_Fourier_matrix(N_harmonics: int = 3) -> np.ndarray:
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

def Fourier_to_Taylor_matrix_2(N_harmonics: int = 3) -> np.ndarray:
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

def Fourier_to_Taylor_matrix(N_harmonics: int = 3) -> np.ndarray:
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

def numpy_to_latex(array: np.ndarray, matrix_type = "bmatrix", precision = 4):
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

def complex_to_rgba(input_array: np.ndarray = np.zeros((2, 2)), phase_shift: float = 0., alpha: float = 1., n_max: int = 1) -> np.ndarray:
    phase = np.angle(input_array) + phase_shift
    hue = (phase + np.pi) / (2 * np.pi) % 1
    mag = np.abs(input_array)
    
    flat_order = np.argsort(mag, axis = None)
    rows, cols = np.unravel_index(flat_order, mag.shape)
    sorted = np.array(list(zip(rows, cols)))
    maximum = np.sum([mag[sorted[-idx, 0], sorted[-idx, 1]] for idx in range(1, n_max + 1)]) / n_max
    for idx in range(1, n_max): mag[sorted[-idx, 0], sorted[-idx, 1]] = maximum
    
    value = mag / maximum # np.max(mag)
    saturation = np.ones_like(hue)
    hsv = np.stack((hue, saturation, value), axis = -1)
    rgb = np.clip(mcolors.hsv_to_rgb(hsv), 0, 1)
    rgba = np.concatenate((rgb, np.full_like(rgb, alpha)), axis = 2)[:, :, :4]
    return rgba

