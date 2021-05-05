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
    
if __name__=='__main__':
    # testing
    import astropy.units as u
    import axion as ax
    import natural_units as nat
    import util_toolkit as util

    N = 2000
    tgrid = np.arange(N)*u.day
    ma = 1.*u.Hz
    gaNN = 1.0e-14*u.GeV**-1
    sigma_noise = (1.*u.aT).to_value(u.T)
    X = np.random.normal(scale=sigma_noise, size=2*N)
    
    print("Computing matrix...")
    util.start_timer()
    C = ax.compute_eff_cov_matrix(ma, tgrid).to_value(u.dimensionless_unscaled)
    util.stop_timer()
    
    g = nat.convert(np.sqrt(ax.rhodm)*gaNN/ax.gammaHe, u.T, value=True)
    Cnoise = np.diag(sigma_noise**2*np.ones(2*len(tgrid)))

    print("Computing Eigenvalues...")
    util.start_timer()
    mat = CovMatrix(C)
    util.stop_timer()

    print("Determinants")
    util.start_timer()
    logdet_full = np.linalg.slogdet(g**2*C + Cnoise)[1]
    util.stop_timer()

    util.start_timer()
    logdet = mat.logdet(g, sigma_noise)
    util.stop_timer()

    """
    print("Inverses")
    util.start_timer()
    inv_full = np.linalg.inv(g**2*C + Cnoise)
    util.stop_timer()
    util.start_timer()
    inv = mat.inv(g, sigma_noise)
    util.stop_timer()
    """
    
    print("Chi2")
    util.start_timer()
    inv_full = np.linalg.inv(g**2*C + Cnoise)
    #chi2_full = np.einsum('i,ij,j', X, inv_full, X)
    chi2_full = X @ inv_full @ X
    util.stop_timer()

    util.start_timer()
    chi2 = mat.chi2(g, sigma_noise, X)
    util.stop_timer()

    print(logdet_full, logdet)
    print(chi2_full, chi2)
