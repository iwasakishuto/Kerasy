# coding: utf-8
import numpy as np
from ._kernel import kernel_handler
from ..utils import basis_transformer
from ..utils import flush_progress_bar

class LinearRegression():
    def __init__(self, basis="none", **basisargs):
        self.basis = basis_transformer(basis, **basisargs)

    def basis_transform(self, X):
        X   = X.reshape(-1,1) if X.ndim==1 else X
        N,_ = X.shape
        X   = self.basis.transform(X)
        D   = int(np.product(X.shape)/N)
        X   = X.reshape(N,D)
        return X

    def fit(self, train_x, train_y):
        """
        @param train_x: shape=(N,?)
        @param train_y: shape=(N,M)
        @val         w: shape=()
        """
        train_x = self.basis_transform(train_x).astype(float)
        self.w = np.linalg.solve(train_x.T.dot(train_x), train_x.T.dot(train_y))

    def predict(self, X):
        """
        @param X: shape=(N',D)
        """
        X = self.basis_transform(X)
        return X.dot(self.w) # shape=(N',M)

class LinearRegressionRidge(LinearRegression):
    def __init__(self, lambda_, basis="none", **basisargs):
        self.lambda_ = lambda_
        super().__init__(basis=basis, **basisargs)

    def fit(self, train_x, train_y):
        """
        @param train_x: shape=(N,?)
        @param train_y: shape=(N,M)
        @val         w: shape=()
        """
        train_x = self.basis_transform(train_x)
        N, M = train_x.shape
        self.w = np.linalg.solve(self.lambda_*np.identity(M) + train_x.T.dot(train_x), train_x.T.dot(train_y))

class LinearRegressionLASSO(LinearRegression):
    def __init__(self, lambda_, basis="none", **basisargs):
        self.lambda_ = lambda_
        super().__init__(basis=basis, **basisargs)

    @staticmethod
    def prox(w,alpha,rho,lambda_):
        z0 = w + alpha/rho
        c = lambda_/rho
        return z0-c if c<z0 else z0+c if z0<-c else 0

    def fit(self, train_x, train_y, rho=1e-3, tol=1e-7, max_iter=100, verbose=1):
        """
        @param train_x: shape=(N,?)
        @param train_y: shape=(N,M)
        @val         w: shape=()
        """
        train_x = self.basis_transform(train_x)
        N, M = train_x.shape
        # Initialization.
        alpha = np.ones(shape=(M))
        z = np.ones(shape=(M))
        for it in range(max_iter):
            w = np.linalg.solve(train_x.T.dot(train_x)+rho*np.identity(M), train_x.T.dot(train_y)-alpha+rho*z)
            z = np.asarray([self.prox(w_,alpha_,rho,self.lambda_) for w_,alpha_ in zip(w,alpha)])
            alpha += rho*(w-z)
            diff = np.sqrt(np.sum(np.square(w-z)))
            if diff < tol: break
            flush_progress_bar(it, max_iter, metrics={"diff": diff}, verbose=verbose)
        if verbose: print()
        self.w = w

class BayesianLinearRegression(LinearRegression):
    def __init__(self, alpha=1, beta=25, basis="none", **basisargs):
        self.alpha=alpha
        self.beta=beta
        self.SN=None
        self.mN=None
        super().__init__(basis=basis, **basisargs)

    def fit(self, train_x, train_y):
        train_x = self.basis_transform(train_x)
        N,M = train_x.shape
        self.SN = np.linalg.inv(self.alpha*np.eye(M) + self.beta*train_x.T.dot(train_x))
        self.mN = self.beta*self.SN.dot(train_x.T.dot(train_y))

    def predict(self, X):
        X = self.basis_transform(X)
        mu = np.array([self.mN.T.dot(x) for x in X])
        std = np.sqrt(np.array([1/self.beta + x.T.dot(self.SN).dot(x) for x in X]))
        return (mu, std)

class EvidenceApproxBayesianRegression(BayesianLinearRegression):
    def __init__(self, alpha=1, beta=1, basis="none", **basisargs):
        super().__init__(alpha=alpha, beta=beta, basis=basis, **basisargs)
        self.gamma = None

    def fit(self, train_x, train_y, max_iter=100):
        train_x_ = self.basis_transform(train_x)
        N,M = train_x_.shape
        Phi = train_x_.T.dot(train_x_)
        eigvals = np.linalg.eigvals(Phi)
        for it in range(max_iter):
            params = [self.alpha, self.beta]
            super().fit(train_x, train_y)
            lambda_ = self.beta * eigvals
            self.gamma = np.sum(lambda_ / (self.alpha + lambda_))
            self.alpha = self.gamma / self.mN.dot(self.mN)
            self.beta  = (N-self.gamma) / np.sum( (train_y-train_x_.dot(self.mN))**2 )
            if np.allclose(params, [self.alpha, self.beta]): break
            flush_progress_bar(it, max_iter, barname="Search for Hyper Parameters (alpha, beta)", verbose=verbose)
        super().fit(train_x, train_y)

    def evidence(self, train_x, train_y):
        """ loglikelihood of marginalization ln p(y|α,β) PRML(3.86) """
        train_x = self.basis_transform(train_x)
        N,M = train_x.shape
        A = self.alpha*np.eye(M) + self.beta*train_x.T.dot(train_x)
        EmN = self.beta/2 * np.sum( (train_y - train_x.dot(self.mN))**2 ) + self.alpha/2 * self.mN.dot(self.mN)
        logmarg = M/2*np.log(self.alpha) + N/2*np.log(self.beta) - EmN - 1/2*np.log(np.linalg.det(A)) - N/2*np.log(2*np.pi)
        return logmarg

class KernelRegression():
    def __init__(self, lamda, kernel="gaussian", **kernelargs):
        self.lamda = lamda
        self.kernel = kernel_handler(kernel, **kernelargs)
        self.x_train = None # shape=(N,D)

    def fit(self, x_train, y_train):
        """
        @param x_train: shape=(N, D)
        @param y_train: shape=(N, M)
        """
        N = x_train.shape[0]
        K = np.asarray([[self.kernel(x, x_prime) for x in x_train] for x_prime in x_train])
        self.theta = np.linalg.solve(K.T.dot(K)+self.lamda*np.identity(N), K.T.dot(y_train))
        self.x_train = x_train

    def predict(self, X):
        """
        @param X: shape=(N', D)
        """
        K = np.asarray([[self.kernel(x, x_prime) for x_prime in self.x_train] for x in X])
        predictions = K.dot(self.theta) # (N',N)@(N,M) = (N',M)
        return predictions
