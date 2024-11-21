import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def hw3_rhs(phi, x, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]

tol = 1e-4 
L = 4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)
epsilon = 0.1
esoln = np.zeros(5)
ysoln = np.zeros((81,5))

for jmodes in range(5):  # begin mode loop
    depsilon = 0.2  # default step size in beta
    for j in range(1000):  # begin convergence loop for beta
        y0 = [1, np.sqrt(4**2 - epsilon)]
        sol = solve_ivp(lambda x, y: hw3_rhs(y, x, epsilon), [x[0], x[-1]], y0, t_eval = x)
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
    

    esoln[jmodes] = epsilon
    norm = np.sqrt(np.trapz(ys[:,0]**2, x))
    ysoln[:, jmodes] = np.abs(ys[:,0] / norm)
    epsilon += 0.2

A1 = ysoln
A2 = esoln

print("Eigenfunction: ", A1)
print("\nEigenvalues: ", A2)
