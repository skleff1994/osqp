# osqp

My own python implementation of Oxford's OSQP solver ( https://osqp.org/docs/solver/index.html ) based on the paper(see `osqp.pdf`). This implementation is obviously not competitive with a thoroughly optimized C++ based solver but I was able to reproduce the convergence characteristics described in the paper using some of the special features (such as solution polishing and step size adaptation). It works fine on MPC for linear inverted pendulum (but takes up to  

# Requirements
- scikit-sparse ( https://scikit-sparse.readthedocs.io/en/latest/index.html ) for sparse Cholesky decomposition

# Usage
See examples against the original OSQP solver `test_simple_qp.py` and `test_hilbert_qp.py`
