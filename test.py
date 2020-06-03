# Author: Sebastien
# My python implementation of the OSQP solver
# The original OSQP solver and the paper are available at : https://osqp.org/docs/index.html

# Test against original OSQP

from solver import QPADMMSolver
from scipy.linalg import hilbert
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

# RANDOM QP
print("----DEFINE RANDOM QP----")
# Define a random problem
P = hilbert(3)
q = np.array([[1, 1, 1]]).T
A = np.eye(3)
lb = -2*np.array([1, 1, 1])
ub = 2*np.array([1, 1, 1])

# ORIGINAL OSQP
print("----SOLVE WITH OSQP (REFERENCE)----")
# Create an OSQP object
prob = osqp.OSQP()
# Setup workspace and change alpha parameter
prob.setup(sparse.csc_matrix(P), q, sparse.csc_matrix(A), lb, ub, alpha=1.6, polish=True,  eps_abs=1e-5, eps_rel=1e-5)
# Solve problem
result = prob.solve()

print("----SOLVE WITH MY SOLVER----")
# Instantiate solver
rho = .1
sigma = 1e-6
alpha = 1.6
solver = QPADMMSolver(P, q, A, lb, ub, rho, sigma, alpha)
# Run solver
obj, x, z, y, res_prim, res_dual, tot_time = solver.solve(maxit=1000, abs_tol=1e-6, rel_tol=1e-6, polish=False, adaptive_rho=True)

## PLOT
    # Objective
obj_ = [float(i) for i in solver.obj_all]
    # Primal residual norm
res_prim_ = [res for res in solver.res_prim_all]
    # Dual residual norm
res_dual_ = [res for res in solver.res_dual_all]
    # Create figs and subplots
fig, ax = plt.subplots(3, 1)
    # Plot objective
ax[0].plot(obj_, 'b-', label='Objective')
ax[0].set(xlabel='Iterations', ylabel='Objective value')
ax[0].legend(loc='upper right', prop={'size': 16})
ax[0].grid()
    # Plot primal res
ax[1].semilogy(res_prim_, 'r-', label='Primal residual norm')
ax[1].set(xlabel='Iterations', ylabel='Primal residual norm value')
ax[1].legend(loc='upper right', prop={'size': 16})
ax[1].grid()
    # Plot dual res
ax[2].semilogy(res_dual_, 'g-', label='Dual residual norm')
ax[2].set(xlabel='Iterations', ylabel='Dual residual norm value')
ax[2].legend(loc='upper right', prop={'size': 16})
ax[2].grid()
    # Custom and save
fig.subplots_adjust(hspace=.5)
fig.suptitle('Objective and residuals for Hilbert (n=3)', size=16)
plt.savefig("hilbert3.png")
plt.show()


