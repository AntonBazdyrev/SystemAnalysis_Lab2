from numpy import linalg
import numpy as np


class InconsistentSystemSolver:
    def __init__(self, method='conjugate'):
        self.method = method if method in ['lstsq', 'conjugate'] else 'conjugate'

    def conjugate_grad(self, A, b, x=None, eps=1e-5):
        """
        Description
        -----------
        Solve a linear equation Ax = b with conjugate gradient method.
        Parameters
        ----------
        A: 2d numpy.array of positive semi-definite (symmetric) matrix
        b: 1d numpy.array
        x: 1d numpy.array of initial point
        Returns
        -------
        1d numpy.array x such that Ax = b
        """
        n = A.shape[1]
        if not x:
            x = np.ones(n)
        r = A @ x - b
        p = - r
        r_k_norm = np.dot(r, r)
        for i in range(2 * n):
            Ap = np.dot(A, p)
            alpha = r_k_norm / np.dot(p, Ap)
            x += alpha * p
            r += alpha * Ap
            r_kplus1_norm = np.dot(r, r)
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm
            if r_kplus1_norm < eps:
                # print ('Itr:', i)
                break
            p = beta * p - r
        return x

    def nonlinear_conjugate_grad(self, A, b, eps=1e-5):
        return self.conjugate_grad(A.T @ A, A.T @ b, eps=eps)

    def lstsq(self, A, b):
        return linalg.lstsq(A, b)[0]

    def solve(self, A, b):
        return self.lstsq(A, b) if self.method == 'lstsq' else self.nonlinear_conjugate_grad(A, b)
