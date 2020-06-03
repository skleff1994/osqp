import numpy as np
from numpy.linalg import norm
from scipy.linalg import ldl
from sksparse.cholmod import cholesky, cholesky_AAt
from scipy import sparse
from scipy.sparse.linalg import inv
import time

class QPADMMSolver:
    '''
    QP solver based on ADMM 
    '''
    
    def __init__(self, P, q, A, lb, ub, rho, sigma, alpha):
        '''
        Initialize the QP : min x^T P x + p^T x 
                            s.t. lb <= Ax <= ub
        '''
        
        ### PROBLEM DATA ###
            # Quadratic cost
        self.P = P
        self.q = q
            # Box constraints
        self.A = A
        self.lb = np.array([lb]).T
        self.ub = np.array([ub]).T
            # Number of variables
        self.n = np.shape(P)[0]
            # Number of constraints
        self.m = np.shape(A)[0]
            # Get eq. / ineq. constraints indices
        self.cstr_ids = range(self.m)
        self.eq_ids = []
        for i in self.cstr_ids:
            if (self.lb[i]==self.ub[i]):
                self.eq_ids.append(i)
        self.ineq_ids = [item for item in self.cstr_ids if item not in self.eq_ids]
        

        ### PARAMETERS ###
             # Penalty matrix rho
        self.rho_bar = rho
        self.rho = self.rho_bar*np.eye(self.m)
        self.inv_rho = np.linalg.inv(self.rho)
        # self.inv_rho = inv(self.rho)
            # Penalty sigma
        self.sigma = sigma
            # Relaxation parameter
        self.alpha = alpha


        ### KKT MATRIX ###
            # Dense KKT matrix
        self.kkt_mat = np.zeros((self.n+self.m, self.n+self.m))
        self.kkt_mat[:self.n,:self.n] = self.P + self.sigma*np.eye(self.n)
        self.kkt_mat[self.n:,:self.n] = self.A
        self.kkt_mat[:self.n,self.n:] = self.A.T
        self.kkt_mat[self.n:,self.n:] = -self.inv_rho
            # Init number of factorizations
        self.nb_factorizations = 0
        self.factorization_time = 0
            # Sparse KKT matrix
        self.kkt_mat = sparse.csc_matrix(self.kkt_mat)
            # Initial factorization 
        self.factor_kkt_mat()


        ### UNFEASIBILITY ###
            # Primal / dual unfeasibility status
        self.primal_unfeas = False
        self.dual_unfeas = False


        ### TO RECORD ITERATES ###
            # Objective
        self.obj_all = []
            # Primal
        self.x_tilde_all = [] 
        self.z_tilde_all = [] 
        self.x_all = [] 
        self.z_all = [] 
            # Dual
        self.y_all = [] 
            # Residuals
        self.res_prim_all = [] 
        self.res_dual_all = []  


    def update_rho(self, update_factor):
        '''
        Update rho (penalty parameter)
        '''
        # Update rho_bar
        self.rho_bar = self.rho_bar*update_factor
        
        # Update rho diag elements depending on constraints
        self.rho[self.eq_ids,self.eq_ids] = 1e3*self.rho_bar
        self.rho[self.ineq_ids,self.ineq_ids] = self.rho_bar

        # Get inverse
        self.inv_rho = np.linalg.inv(self.rho)
        # self.inv_rho = inv(self.rho)

        # Re-factor KKT mat
        self.factor_kkt_mat()


    def factor_kkt_mat(self):
        '''
        Sparse Cholesky factorization of KKT matrix
        '''
        # Measure factorization time
        factorization_start_time = time.time()

        # Update KKT matrix with new rho
        self.kkt_mat[self.n:,self.n:] = -self.inv_rho

        # Make it sparse and decompose        
        self.factor = cholesky(self.kkt_mat, mode="simplicial")

        # Increment factorizations counter
        self.nb_factorizations += 1

        # Update factorization time
        self.factorization_time = time.time() - factorization_start_time


    def obj_fun(self, x):
        '''
        Returns original objective function's value
        '''
        # Compute objective
        f = .5*x.T.dot(self.P).dot(x) + self.q.T.dot(x)

        return f


    def get_residuals(self, xk, zk, yk, abs_tol, rel_tol):
        '''
        Compute the current primal and dual residuals + tolerances
        '''
        # Primal residual
        res_prim = norm(self.A.dot(xk) - zk, np.inf)
         
        # Dual residual
        res_dual = norm(self.P.dot(xk) + self.q + self.A.T.dot(yk), np.inf)

        # Stopping criterion on residuals
        a = max(norm(self.A.dot(xk), np.inf), norm(zk, np.inf))
        b = max(norm(self.P.dot(xk), np.inf), norm(self.A.T.dot(yk), np.inf), norm(self.q, np.inf))
        eps_prim = abs_tol + rel_tol*a
        eps_dual = abs_tol + rel_tol*b

        # Get rho update factor for later
        if (a!=0) and (b!=0):
            rho_update_factor = np.sqrt((res_prim/a)/(res_dual/b))
        else:
            rho_update_factor = 1

        return res_prim, res_dual, eps_prim, eps_dual, rho_update_factor
        

    def admm_step(self, xk, zk, yk):
        '''
        Compute next iterates of the ADMM algorithm
        '''
        # RHS of the KKT system
        kkt_rhs = np.zeros((self.m+self.n,1))
        kkt_rhs[:self.n] = self.sigma*xk - self.q
        kkt_rhs[self.n:] = zk - self.inv_rho.dot(yk)

        # Sovle KKT system with sparse Cholesky factorization
        xtilde_nu_kp1 = self.factor(kkt_rhs)
        
        # Get x(k+1),nu(k+1)
        xtilde = xtilde_nu_kp1[:self.n]
        nukp1 = xtilde_nu_kp1[self.n:]
        xkp1 = self.alpha*xtilde + (1-self.alpha)*xk
        
        # Get z(k+1)
        ztilde = zk + self.inv_rho.dot(nukp1 - yk) 
        vkp1 = self.alpha*ztilde + (1-self.alpha)*zk + self.inv_rho.dot(yk)
        zkp1 = np.clip(vkp1, self.lb, self.ub)

        # Get y(k+1)
        ykp1 = self.rho.dot(vkp1 - zkp1) 

        return xkp1, xtilde, zkp1, ztilde, ykp1


    def solve(self, maxit=1000, abs_tol=1e-3, rel_tol=1e-3, adaptive_rho=False, polish=False, eps_inf=1e-4):
        ''' 
        Runs ADMM algorithm
        Returns sequences of iterates (objective, primal, dual + residuals)
        + misc. logs
        '''

        # Start time
        start = time.time()

        # Initialize iterates
        xk = np.zeros((self.n,1))
        zk = np.ones((self.m,1))
        yk = np.zeros((self.m,1))
        xtilde = xk
        ztilde = zk
        
        # Iterations counter
        k = 0

        # While not > maxit
        while k<maxit:
            
            # Get objective value at current iterate (xk,zk,yk)
            prim_obj = self.obj_fun(xk)

            # Record current objective value
            self.obj_all.append(prim_obj)

            # Record iterates
            self.x_all.append(xk) 
            self.z_all.append(zk) 
            self.x_tilde_all.append(xtilde) 
            self.z_tilde_all.append(ztilde) 
            self.y_all.append(yk) 

            # Get residuals and rho_bar update factor
            res_prim, res_dual, eps_prim, eps_dual, rho_update_factor = self.get_residuals(xk, zk, yk, abs_tol, rel_tol)

            # Record residuals
            self.res_prim_all.append(res_prim)
            self.res_dual_all.append(res_dual) 

            # Check if termination criteria is satisfied
            if res_prim<=eps_prim and res_dual<=eps_dual:
                break
                
            # Otherwise perform the ADMM iteration
            else:

                # Get new iterates
                xkp1, xtilde_kp1, zkp1, ztilde_kp1, ykp1 = self.admm_step(xk, zk, yk)

                # Check primal and dual unfeasibility
                self.is_unfeasible(xkp1 - xk, zkp1 - zk, ykp1 - yk)

                # If primal or dual unfeasible stop there
                if(self.primal_unfeas) or (self.dual_unfeas):
                    break

                # Otherwise update the iterates
                else:

                    # Update iterates (xk,zk,yk) and (xtilde,ztilde)
                    xk = xkp1
                    xtilde = xtilde_kp1
                    zk = zkp1
                    ztilde = ztilde_kp1
                    yk = ykp1

                    # Update counter 
                    k = k+1

                    # Rho update
                    if adaptive_rho:
                        # If not worth it, pass
                        if (0.2<=rho_update_factor and rho_update_factor<=5):
                            pass
                        # If new rho_bar is sufficiently different from old one
                        else:
                            # And if re-factoring the KKT matrix is not too expensive
                            if time.time() - start >= .4*self.factorization_time:
                                self.update_rho(rho_update_factor)

        # Record computation time
        tot_time = time.time() - start

        # If unfeasible return unfeas
        if(self.primal_unfeas):
            print("Primal unfeasible !")
            return None

        if(self.dual_unfeas):
            print("Dual unfeasible !")
            return None

        # Polish 
        if polish:
            self.polish(xk, xtilde, zk, ztilde, yk, abs_tol, rel_tol)

        # Extract final data
        obj_opt = float(self.obj_all[-1])
        res_prim_opt = self.res_prim_all[-1]
        res_dual_opt = self.res_dual_all[-1]
        x_opt = self.x_all[-1]
        z_opt = self.z_all[-1]
        y_opt = self.y_all[-1]

        # Display
        print("################################")
        print("Converged in ", k, "iterations")
        print("  Objective : ", obj_opt)
        print("    prim res. ", res_prim_opt)
        print("    dual res. ", res_dual_opt)
        print("  Rho = ", self.rho_bar, "(",self.nb_factorizations," updates ) | Sigma = ", self.sigma, " | Alpha = ", self.alpha)
        print("  Time =", tot_time,"s")
        print("################################")

        return obj_opt, x_opt, z_opt, y_opt, res_prim_opt, res_dual_opt, tot_time
        
    
    def polish(self, x, xtilde, z, ztilde, y, abs_tol=1e-3, rel_tol=1e-3, delta=1e-6, maxit=100):
        '''
        Solution polishing through iterative refinement
        '''
        # Define the set of active constraints for upper and lower bounds
        L = list(np.where(y<0)[0])
        U = list(np.where(y>0)[0])

        # Subvectors l_L and u_U
        l_L =  self.lb[L] 
        u_U =  self.ub[U] 
        
        # Select rows of A with index in L
        A_L = self.A[L,:]
        A_U = self.A[U,:]

        # Construct K
            # Dimensions
        N = self.n + np.shape(A_L)[0] + np.shape(A_U)[0]
        K = np.zeros((N, N))
            # Fill 
        K[:self.n, :self.n] = self.P
            # Upper right
        K[:self.n, self.n:self.n + np.shape(A_L)[0]] = A_L.T
        K[:self.n, N-np.shape(A_U)[0]:N] = A_U.T
            # Lower left
        K[self.n:self.n + np.shape(A_L)[0], :self.n] = A_L
        K[N-np.shape(A_U)[0]:N, :self.n] = A_U

        # Regularize K
        DK = np.zeros(np.shape(K))
        DK[:self.n, :self.n] = delta*np.eye(self.n)
        DK[self.n:self.n + np.shape(A_L)[0], self.n:self.n + np.shape(A_L)[0]] = -delta*np.eye(np.shape(A_L)[0])
        DK[N-np.shape(A_U)[0]:N, N-np.shape(A_U)[0]:N] = -delta*np.eye(np.shape(A_U)[0])

        # RHS and unknown
        g = np.array([np.concatenate((-self.q, l_L, u_U), axis=None)]).T
        tk = np.zeros(np.shape(g)) 

        # Iterative refinement
        i=0
        while(i<=maxit):
            dtk = np.linalg.solve(K+DK, g-K.dot(tk))
            if(norm(dtk,np.inf)==0):
                break
            else:
                tk = tk + dtk
                i = i+1
        print(i)
        # Extract polished y from tk
        tmp = tk[self.n:]
        y_pol = np.zeros(np.shape(y)) 
        y_pol[L] = tmp[:len(L)]
        y_pol[U] = tmp[len(L):]

        # Extract polished x and z
        x_pol = tk[:self.n]
        z_pol = self.A.dot(x_pol)
        
        # Compute improved objective
        obj_pol = self.obj_fun(x_pol)

        # Get corresponding residuals
        res_prim_pol, res_dual_pol, eps_prim_pol, eps_dual_pol, update_factor = self.get_residuals(x_pol, z_pol, y_pol, abs_tol, rel_tol)
        
        # Record refined iterates
        self.res_prim_all.append(res_prim_pol)
        self.res_dual_all.append(res_dual_pol) 
        self.x_all.append(x_pol)
        self.z_all.append(z_pol) 
        self.x_tilde_all.append(xtilde)
        self.z_tilde_all.append(ztilde)
        self.y_all.append(y_pol)
        self.obj_all.append(obj_pol)
    
    def is_unfeasible(self, dxk, dzk, dyk, eps_inf=1e-4):
        '''
        Check if the problem is primal or dual infeasible
        '''
        # Get dyk+, dyk- 
        yp = np.maximum(dyk, 0)
        ym = np.minimum(dyk, 0)

        # Primal unfeasibility
        tol_prim = eps_inf*norm(dyk,np.inf)
        if (norm(self.A.T.dot(dyk),np.inf)<= tol_prim) and (self.ub.T.dot(yp)+self.lb.T.dot(ym)<=tol_prim):
            self.primal_unfeas = True

        # Dual unfeasibility
        tol_dual = eps_inf*norm(dxk,np.inf)
        if (norm(self.P.dot(dxk),np.inf)<=tol_dual) and (self.q.T.dot(dxk)<=tol_dual) and (np.abs(self.A.dot(dxk).all())<=tol_dual):
            self.dual_unfeas = True
    