# https://www.geeksforgeeks.org/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/

import numpy as np

tol = 1e-6
max_iter = 100
lam_prev = 0

A=np.random.rand(3,3)
A=A+A.T
x=np.random.rand(3)

[evals,evecs] = np.linalg.eig(A)
print(evals[0])
print(evecs[:,0])

for i in range(max_iter):
   x = A @ x / np.linalg.norm(A @ x)
   lam = (x.T @ A @ x) / (x.T @ x)
   if np.abs(lam - lam_prev) < tol:
      break
   lam_prev = lam
   
print(lam)
print(x)
