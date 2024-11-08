import numpy as np
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp

############################################################################################################
# part a
def hw2_rhs(phi, x, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]

tol = 1e-4 
L = 4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)
epsilon = 0.1
esola = np.zeros(5)
ysola = np.zeros((81,5))

for jmodes in range(5):  # begin mode loop
    depsilon = 0.2  # default step size in beta
    for j in range(1000):  # begin convergence loop for beta
        y0 = [1, np.sqrt(4**2 - epsilon)]
        sol = solve_ivp(lambda x, y: hw2_rhs(y, x, epsilon), [x[0], x[-1]], y0, t_eval = x)
        ys = sol.y.T
        bc = ys[-1,1] + np.sqrt(L**2 - epsilon) * ys[-1,0]

        if abs(bc) < tol:
            # print(j)
            break

        if (-1)**jmodes * bc > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2
    

    esola[jmodes] = epsilon
    norm = np.sqrt(np.trapz(ys[:,0]**2, x))
    ysola[:, jmodes] = np.abs(ys[:,0] / norm)
    epsilon += 0.2

A1 = ysola
A2 = esola

############################################################################################################
# part b
L = 4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)

dx = x[1] - x[0]
A = np.zeros((n-2, n-2))

for j in range(n-2):
    A[j, j] = -2 - (dx**2) * x[j+1]**2
    if j < n-3:
        A[j+1, j] = 1
        A[j, j+1] = 1
A[0,0] = A[0,0] + 4/3
A[0,1] = A[0,1] - 1/3
A[-1,-1] = A[-1,-1] + 4/3
A[-1,-2] = A[-1,-2] - 1/3

eigval, eigvecs = eigs(-A, k = 5, which = 'SM')
v2 = np.vstack([(4/3)*eigvecs[0,:]-(1/3)*eigvecs[1,:], eigvecs, (4/3)*eigvecs[-1,:]-(1/3)*eigvecs[-2,:]])

ysolb = np.zeros((n, 5))
esolb = np.zeros(5)

for j in range(5):
    norm = np.sqrt(np.trapz(v2[:,j]**2, x))
    ysolb[:,j] = abs(v2[:,j] / norm)

esolb = eigval[:5] / dx**2

A3 = ysolb
A4 = esolb

############################################################################################################
# part c
def hw3_rhs_c(x, y, epsilon, gamma):
    return [y[1], (gamma * (y[0])**2 + x**2 - epsilon) * y[0]]

L = 2
xp = [-L, L]
x = np.arange(-L, L, 0.1)
n = len(x)

eigvals = []
eigfuncs = []

Esolcpos, Esolcneg = np.zeros(2), np.zeros(2)
ysolcpos, ysolcneg = np.zeros((n, 2)), np.zeros((n, 2))


for gamma in [0.05, -0.05]:
    tol = 1e-6
    EO = 0.1
    for jmodes in range(2):  # begin mode loop
        E = EO
        dE = 0.2
        A = 1e-3
        for j in range(1000):  
            y0 = [A, np.sqrt(L**2 - E) * A]
            ys = solve_ivp(hw3_rhs_c, xp, y0, t_eval = x, args=(E, gamma))
            norm = np.trapz(ys.y[0,:] * ys.y[0,:], x)
            y1 = np.sqrt(L**2 - E) * ys.y[0,-1]
            y2 = ys.y[1,-1]

            if (abs(y1 + y2) < tol) and (abs(1 - norm) < tol):
                break
            else:
                A = A / np.sqrt(norm)

            ys = solve_ivp(lambda x, y: hw3_rhs_c(x, y, E, gamma), [x[0], x[-1]], y0, t_eval = x)
            norm = np.trapz(ys.y[0,:] * ys.y[0,:], x)
            y1 = np.sqrt(L**2 - E) * ys.y[0,-1]
            y2 = ys.y[1,-1]
            if abs(y1 + y2) < tol and abs(1 - norm) < tol:
                break
            if (-1)**jmodes * (y1 + y2) > 0:
                E += dE
            else:
                E -= dE / 2
                dE /= 2
        eigvals.append(E)
        eigfuncs.append(abs(ys.y[0,:]))
        EO = E + 0.1

A5 = np.array(eigfuncs).T[:, :2]
A6 = np.array(eigvals)[:2]
A7 = np.array(eigfuncs).T[:, 2:]
A8 = np.array(eigvals)[2:]

############################################################################################################
# part d
def hw1_rhs_a(x, y, E):
    return [y[1], (x**2-E)*y[0]]

L = 2 
x_span = [-L, L]
E = 1
A = 1
y0 = [A, np.sqrt(L**2-E)*A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
dt45, dt23, dtRadau, dtBDF = [], [], [], []

for tol in tols:
    options = {'rtol':tol, 'atol':tol}
    sol45 = solve_ivp(hw1_rhs_a, x_span, y0, method = 'RK45', args = (E,), **options)
    sol23 = solve_ivp(hw1_rhs_a, x_span, y0, method = 'RK23', args = (E,), **options)
    solRadau = solve_ivp(hw1_rhs_a, x_span, y0, method = 'Radau', args = (E,), **options)
    solBDF = solve_ivp(hw1_rhs_a, x_span, y0, method = 'BDF', args = (E,), **options)

    # calculate average time steps
    # for each method, add mean(diff(sol.t))
    dt45.append(np.mean(np.diff(sol45.t)))
    dt23.append(np.mean(np.diff(sol23.t)))
    dtRadau.append(np.mean(np.diff(solRadau.t)))
    dtBDF.append(np.mean(np.diff(solBDF.t)))

# perform linear regression (log-log) to determine slopes
fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fitRadau = np.polyfit(np.log(dtRadau), np.log(tols), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)
#   extract slopes
#       slopes are the first [0] value of fit45, fit23, fitRadau, fitBDF

A9 = [fit45[0], fit23[0], fitRadau[0], fitBDF[0]]


############################################################################################################
#part e
h = np.array([np.ones_like(x), 2*x, 4*(x**2)-2, 8*(x**3)-12*x, 16*(x**4)-48*(x**2)+12])
phi = np.zeros((len(x),5))

for j in range(5):
    phi[:,j] = np.exp(-(x**2)/2)*h[j,:] / np.sqrt(np.math.factorial(5)*(2**j)*np.sqrt(np.pi)).T

erpsi_a = np.zeros(5)
erpsi_b = np.zeros(5)
er_a = np.zeros(5)
er_b = np.zeros(5)

for j in range(5):
    erpsi_a[j] = np.trapz((abs(ysola[:,j])-abs(phi[:,j]))**2, x)
    erpsi_b[j] = np.trapz((abs(ysolb[:,j])-abs(phi[:,j]))**2, x)
    er_a[j] = 100 * abs(esola[j]-((2*(j+1)-1)/(2*(j+1)-1)))
    er_b[j] = 100 * abs(esolb[j]-((2*(j+1)-1)/(2*(j+1)-1)))

A10 = erpsi_a
A12 = erpsi_b
A11 = er_a
A13 = er_b








            
    








