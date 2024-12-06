import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
from scipy.linalg import lu, solve_triangular
import imageio

############################################################################################################
# part a
tspan = np.arange(0, 4 + 0.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w = np.exp(- X**2 - (1/20)*(Y**2)) + np.zeros((nx, ny))
w2 = w.reshape(N)

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

m = 64   # N value in x and y directions
n = m * m  # total size of matrix
x = np.linspace(-10,10,m+1)
x = x[:m]
dx = x[1] - x[0]

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonalsA = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsetsA = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
A = spdiags(diagonalsA, offsetsA, n, n).toarray() / dx**2
A[0,0] = 2/dx**2

diagonalsB = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsetsB= [-(n-m), -m, m, n-m]
B = spdiags(diagonalsB, offsetsB, n, n).toarray() / (2*dx)

diagonalsC = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsetsC = [-m+1, -1, 1, m-1]
C = spdiags(diagonalsC, offsetsC, n, n).toarray() / (2*dx)

def spc_rhsA(t, wt2, nx, ny, N, KX, KY, K , nu):
    w = wt2.reshape((nx, ny))
    wt = fft2(w)
    psit = -wt / K
    psi = np.real(ifft2(psit)).reshape(N)
    rhs = nu * (A @ wt2) - (B @ psi) * (C @ wt2) + (B @ wt2) * (C @ psi)
    return rhs

wsolA = solve_ivp(spc_rhsA, [0, 4], w2, t_eval=tspan, args=(nx, ny, N, KX, KY, K, nu), method='RK45')
A1 = wsolA.y

############################################################################################################
# part b
A[0,0] = 2

def spc_rhsB1(t, wt2, A, B, C, nu):
    psi = np.linalg.solve(A, wt2)
    rhs = nu * (A @ wt2) - (B @ psi) * (C @ wt2) + (B @ wt2) * (C @ psi)
    return rhs

wsolB1 = solve_ivp(spc_rhsB1, [0, 4], w2, t_eval=tspan, args=(A, B, C, nu))
A2 = wsolB1.y

P, L, U = lu(A)

def spc_rhsB2(t, wt2, P, L, U, B, C, nu):
    Pb = np.dot(P, wt2)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    rhs = nu * (A @ wt2) - (B @ psi) * (C @ wt2) + (B @ wt2) * (C @ psi)
    return rhs

wsolB2 = solve_ivp(spc_rhsB2, [0, 4], w2, t_eval=tspan, args=(P, L, U, B, C, nu))
A3 = wsolB2.y

# for j, t in enumerate(tspan):
#     w = wsolB3.y[:N, j].reshape((nx, ny))
#     plt.subplot(3, 3, j + 1)
#     plt.pcolor(x, y, w, cmap='RdBu_r')
#     plt.title(f'Time: {t}')
#     plt.colorbar()

# plt.tight_layout
# plt.show()

# frames = []
# # make animation
# for j, t in enumerate(tspan):
#     w = wsolB2.y[:, j].reshape((nx, ny)) # Reconstruct the solution at time t
#     plt.pcolor(x, y, w)
#     plt.title(f'Time: {t}')
#     plt.colorbar()

#     frame_filename = f"frame_{j}.png"
#     plt.savefig(frame_filename)
#     plt.close()
#     frames.append(imageio.imread(frame_filename))
#     # os.remove(frame_filename)

# gif_filename = "same_charged_gaussian.gif"
# imageio.mimsave(gif_filename, frames, fps=3)