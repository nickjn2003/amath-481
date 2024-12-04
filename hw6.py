import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import kron

############################################################################################################
# part a
D1 = D2 = 0.1
beta = 1
T = 4
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

m = 1
u = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(x+1j * Y) - np.sqrt(X**2 + Y**2))
v = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(x+1j * Y) - np.sqrt(X**2 + Y**2))

ut = fft2(u)
vt = fft2(v)
uvt0 = np.hstack([(ut.reshape(N)), (vt.reshape(N))])

def spectral_rhs(t, uvt):
    utc = uvt[0:N]
    vtc = uvt[N:]
    
    ut = utc.reshape(nx, ny)
    vt = vtc.reshape(nx, ny)

    u = ifft2(ut)
    v = ifft2(vt)
    
    A = u * u + v * v
    lam = 1-A
    omeg = - beta * A

    rhs_u = (-D1 * K * ut + fft2(lam * u - omeg * v)).reshape(N)
    rhs_v = (-D2 * K * vt + fft2(omeg * u + lam * v)).reshape(N)
    rhs = np.hstack([(rhs_u), (rhs_v)])
    return rhs

uvtsol = solve_ivp(spectral_rhs, [0, T], uvt0, t_eval=tspan, args=(), method="RK45")
z = uvtsol.y
A1 = z

# for j, t in enumerate(tspan):
#     u = np.real(ifft2(z[0:N], j).reshape((nx, ny)))
#     plt.subplot(3, 3, j+1)
#     plt.pcolor(x, y, u, cmap='RdBu_r')
#     plt.title(f'Time: {t}')
#     plt.colorbar
# plt.tight_layout
# plt.show()

############################################################################################################
# part b
def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	return D, x.reshape(N+1)

N = 30
D, x = cheb(N)
D[N,:] = 0
D[0,:] = 0
D_xx = np.dot(D,D) / 100
y = x
N2 = (N+1) * (N+1)
I = np.eye(len(D_xx))
L = kron(I, D_xx) + kron(D_xx, I)
X, Y = np.meshgrid(x, y)
X = X * 10
Y = Y * 10

m = 1
u = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(x+1j * Y) - np.sqrt(X**2 + Y**2))
v = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(x+1j * Y) - np.sqrt(X**2 + Y**2))
uv0 = np.hstack([(u.reshape(N2)), (v.reshape(N2))])

def RD_2D(t, uv):
    u = uv[0:N2]
    v = uv[N2:]
	
    A_2 = u**2 + v**2
    lam = 1-A_2
    omeg = -beta * A_2

    rhs_u = D1 * np.dot(L, u) + lam * u - omeg * v
    rhs_v = D2 * np.dot(L, u) + omeg * u - lam * v
    rhs = np.hstack([rhs_u, rhs_v])
    return rhs
uvsol = solve_ivp(RD_2D, [0, T], uv0, t_eval=tspan, args=(), method="RK45")
A2 = uvsol.y
print(A2)