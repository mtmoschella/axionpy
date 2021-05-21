import numpy as np

class CovMatrix:
    """
    Efficient matrix computation using eigenvalue/eigenvector diagonalization
    """
    def __init__(self, mat):
        """
        mat: (N,N) covariance matrix in dimensionless units
        """
        L, U = np.linalg.eigh(mat)
        self.L = L
        self.U = U
        self.Uinv = np.transpose(U)

        
        # check if matrix is positive semi-definite
        positive = np.all(self.L>=0.)
        if not positive:
            # repair if all negative eigenvalues are smaller than 1.e-10*largest_eigenvalue
            if np.absolute(np.amin(self.L))<1.e-10*np.amax(self.L):
                self.L[self.L<0.]= 0.
            else:
                raise Exception("ERROR: cov matrix not positive semi-definite")

    def diag(self, g, sigma_noise):
        """
        g: coupling (in T not astropy)
        sigma_noise: noise amplitude (in T not astropy)
        """
        return self.L*g**2 + sigma_noise**2
        
    def __call__(self, g, sigma_noise):
        """
        g: coupling (in T not astropy)
        sigma_noise: noise amplitude (in T not astropy)
        """
        D = self.diag(g, sigma_noise)
        return self.U @ np.diag(D) @ self.Uinv

    def inv(self, g, sigma_noise):
        """
        g: coupling (in T not astropy)
        sigma_noise: noise amplitude (in T not astropy)
        """
        import time
        D = self.diag(g, sigma_noise)
        return self.U @ np.diag(1./D) @ self.Uinv

    def logdet(self, g, sigma_noise):
        """
        g: coupling (in T not astropy)
        sigma_noise: noise amplitude (in T not astropy)
        """        
        D = self.diag(g, sigma_noise)
        return np.sum(np.log(D))

    def chi2(self, g, sigma_noise, X):
        """
        g: coupling (in T not astropy)
        sigma_noise: noise amplitude (in T not astropy)
        X: data (in T not astropy)
        """
        D = self.diag(g, sigma_noise)
        UX = self.Uinv @ X
        return UX/D @ UX
