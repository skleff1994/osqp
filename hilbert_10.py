from solver import QPADMMSolver
from scipy.linalg import hilbert
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt


# RANDOM QP
print("----DEFINE RANDOM QP----")
# Number of decision variables
n = 50
# Number of constraints
m = 30
# Problem data
P = hilbert(n)
q = np.random.randn(n,1)
A = np.random.randn(n,n)
cstr = np.random.randn(n)
lb = -abs(cstr) 
ub = +abs(cstr) 


# ORIGINAL OSQP
print("----SOLVE WITH OSQP (REFERENCE)----")
# Create an OSQP object
prob = osqp.OSQP()
# Setup workspace and change alpha parameter
prob.setup(sparse.csc_matrix(P), q, sparse.csc_matrix(A), lb, ub, alpha=1.6, polish=True, eps_abs=1e-5, eps_rel=1e-5)
# Solve problem
result = prob.solve()


# MY SOLVER
print("----SOLVE WITH MY SOLVER----")

# RUN with different params
polish_ = [False, True, False, True] 
rho_ = [False, False, True, True] 
file_ = ['','polish', 'rho', 'polish_rho']

for i in range(4):

    print("  >> POLISHING : "+str(polish_[i])+" | ADAPTIVE RHO : "+str(rho_[i]))

    # Instantiate solver
    rho = .1
    sigma = 1e-6
    alpha = 1.6
    solver = QPADMMSolver(P, q, A, lb, ub, rho, sigma, alpha)

    # Solve
    obj, x, z, y, res_prim, res_dual, tot_time = solver.solve(maxit=1000, abs_tol=1e-5, rel_tol=1e-5, polish=polish_[i], adaptive_rho=rho_[i])

    # Plot
    # Objective
    obj_ = [float(i) for i in solver.obj_all]
    obj_val = [result.info.obj_val for i in obj_]
    res_prim_ = [res for res in solver.res_prim_all]
    res_dual_ = [res for res in solver.res_dual_all]
    fig, ax = plt.subplots(3, 1)

        # Plot objective
    ax[0].plot(obj_, 'b-', label='Objective')
    ax[0].plot(obj_val, 'k-.', label='Reference (OSQP)')
    ax[0].set(xlabel='Iterations', ylabel='Error')
    ax[0].legend(loc='lower left', prop={'size': 16})
    ax[0].grid()
        # Plot primal res
    ax[1].semilogy(res_prim_, 'r-', label='Primal residual norm')
    ax[1].set(xlabel='Iterations', ylabel='Norm')
    ax[1].legend(loc='lower left', prop={'size': 16})
    ax[1].grid()
        # Plot dual res
    ax[2].semilogy(res_dual_, 'g-', label='Dual residual norm')
    ax[2].set(xlabel='Iterations', ylabel='Norm')
    ax[2].legend(loc='lower left', prop={'size': 16})
    ax[2].grid()
        # Custom and save
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('Objective and residuals for random QP (n=50,m=30)', size=16)

    # Plot
    # Objective
    obj_ = [abs(float(i)-result.info.obj_val) for i in solver.obj_all]
    # obj_val = [result.info.obj_val for i in obj_]
    fig, ax = plt.subplots(3, 1)
        # Plot objective
    ax[0].semilogy(obj_, 'b-', label='Objective')
    ax[0].set(xlabel='Iterations', ylabel='Error')
    ax[0].legend(loc='lower left', prop={'size': 16})
    ax[0].grid()
        # Plot primal res
    ax[1].semilogy(res_prim_, 'r-', label='Primal residual norm')
    ax[1].set(xlabel='Iterations', ylabel='Norm')
    ax[1].legend(loc='lower left', prop={'size': 16})
    ax[1].grid()
        # Plot dual res
    ax[2].semilogy(res_dual_, 'g-', label='Dual residual norm')
    ax[2].set(xlabel='Iterations', ylabel='Norm')
    ax[2].legend(loc='lower left', prop={'size': 16})
    ax[2].grid()
        # Custom and save
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('Objective and residuals for random QP (n=50,m=30)', size=16)
    plt.show()
