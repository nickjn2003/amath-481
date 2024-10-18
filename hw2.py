import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot2(phi, x, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
n0 = 0.1; A = 1; xp = [-4, 4] 
xshoot =  np.linspace(xp[0], xp[1], 81)
A1 = np.zeros((81, 5))
A2 = []

epsilon_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    epsilon = epsilon_start  # initial value of eigenvalue beta
    depsilon = 0.2  # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        x0 = [1, np.sqrt(4**2 - epsilon)]
        y = odeint(shoot2, x0, xshoot, args=(epsilon, )) 
        # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta)) 

        if abs(y[-1, 1] + np.sqrt(4**2 - epsilon) * y[-1, 0]) < tol:  # check for convergence
            A2.append(epsilon)
            # print(epsilon)  # write out eigenvalue
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(4**2 - epsilon) * y[-1, 0]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    epsilon_start = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(np.abs(y[:, 0]) * np.abs(y[:, 0]), xshoot)
    A1[:, modes - 1] = np.abs(y[:, 0] / np.sqrt(norm))
    plt.plot(xshoot, y[:, 0] / np.sqrt(norm), col[modes - 1], label=f'ϕ{modes}')  # plot modes

print("Eigenfunction: ", A1)
print("\nEigenvalues: ", A2)

plt.legend()
plt.show()