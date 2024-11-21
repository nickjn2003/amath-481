import numpy as np
from scipy.sparse.linalg import eigs

# Parameters
L = 4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)
dx = x[1] - x[0]

# Construct the finite-difference matrix with adjusted formulation
A = np.zeros((n - 2, n - 2))
for j in range(n - 2):
    A[j, j] = -2.0 / dx**2 + (x[j + 1]**2)  # Main diagonal with potential term
    if j < n - 3:
        A[j + 1, j] = 1.0 / dx**2  # Lower diagonal
        A[j, j + 1] = 1.0 / dx**2  # Upper diagonal

# Applying boundary condition adjustments
A[0, 0] += 4.0 / 3.0 * (1.0 / dx**2)
A[0, 1] -= 1.0 / 3.0 * (1.0 / dx**2)
A[-1, -1] += 4.0 / 3.0 * (1.0 / dx**2)
A[-1, -2] -= 1.0 / 3.0 * (1.0 / dx**2)

# Calculate eigenvalues and eigenvectors
eigval, eigvecs = eigs(-A, k=5, which='SM')

# Reconstruct full wavefunctions
v2 = np.vstack([
    (4 / 3) * eigvecs[0, :] - (1 / 3) * eigvecs[1, :],
    eigvecs,
    (4 / 3) * eigvecs[-1, :] - (1 / 3) * eigvecs[-2, :]
])

# Normalize each eigenfunction and compute absolute values
ysoln = np.zeros((n, 5))
for j in range(5):
    norm = np.sqrt(np.trapz(v2[:, j]**2, x))
    ysoln[:, j] = np.abs(v2[:, j] / norm)

# Eigenvalues should be in the correct format without extra scaling
esoln = eigval[:5].real

# Output results
print("Eigenfunctions (absolute values):", ysoln)
print("Eigenvalues:", esoln)
