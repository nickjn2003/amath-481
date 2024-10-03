import numpy as np

# Newton-Raphson Method
A1 = np.array([-1.6])
newt_iter = 0
for j in range(1000):
    A1 = np.append(A1, A1[j] - (A1[j] * np.sin(3*A1[j]) - np.exp(A1[j])) / (np.sin(3*A1[j]) + 3*A1[j]*np.cos(3*A1[j]) - np.exp(A1[j])))
    fc = A1[j] * np.sin(3 * A1[j]) - np.exp(A1[j])
    newt_iter += 1

    if abs(fc) < 1e-6:
        break

# Bisection Method
A2 = []
xr = -0.4
xl = -0.7
bi_iter = 0
for j in range(0, 100):
    xc = (xr + xl)/2
    A2.append(xc)
    fc = xc * np.sin(3*xc) - np.exp(xc)
    if (fc > 0):
        xl = xc
    else:
        xr = xc
    bi_iter += 1

    if (abs(fc) < 1e-6):
        break
A2 = np.array(A2)

# Number of iterations for both methods
A3 = np.array([newt_iter, bi_iter])

# Matrices
A = np.array([[1,2], [-1,1]])
B = np.array([[2,0], [0,2]])
C = np.array([[2,0,-3], [0,0,-1]])
D = np.array([[1,2], [2,3], [-1,0]])
x = np.array([1,0]).T
y = np.array([0,1]).T
z = np.array([1,2,-1]).T

A4 = A + B
A5 = 3*x - 4*y
A6 = np.dot(A, x)
A7 = np.dot(B, x-y)
A8 = np.dot(D, x)
A9 = np.dot(D, y) + z
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)