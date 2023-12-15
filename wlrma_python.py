import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataGen:
    def __init__(self, n, p, r, sigma):
        self.n = n
        self.p = p
        self.r = r
        self.sigma = sigma

    def generate_data(self, seed=1):
        np.random.seed(seed)
        # Generate matrices A and B from standard normal distribution
        A = np.random.randn(self.n, self.r)
        B = np.random.randn(self.p, self.r)

        # Compute M = AB^T + E
        M = np.dot(A, B.T) + np.random.normal(0, self.sigma, (self.n, self.p))

        # Generate the matrix of weights W from a uniform distribution
        W = np.random.uniform(0, 1, (self.n, self.p))

        return M, W


class WLRMASolver:
    def __init__(self, M, W, max_iter=300, tol=1e-8,
                 relaxation=False, k=None, lamda=None,
                 solver='baseline', memory=None):
        # Take the inputs and initialize the variables.
        self.M = M
        self.W = W
        self.max_iter = max_iter
        self.tol = tol
        # specify the type of the problem.
        self.relaxation = relaxation
        if self.relaxation:
            self.lamda = lamda
        else:
            self.k = k
        # specify the solver.
        if solver not in ['baseline', 'nesterov', 'anderson']:
            raise ValueError("Invalid solver. Allowed options are: 'baseline', 'nesterov', 'anderson'.")
        self.solver = solver
        if self.solver == 'anderson':
            self.memory = memory
        # Initialize the variables for loss tracking.
        self.losses = []
        self.delta = []

    @staticmethod
    def svd_k(X, k):
        # Implement the k-rank approximation through SVD.
        U, D, V = np.linalg.svd(X)
        return U[:, :k] @ np.diag(D[:k]) @ V[:k, :]

    @staticmethod
    def svd_threshold(X, lamda):
        # Implement the thresholding operator through SVD.
        U, D, V = np.linalg.svd(X)
        return U[:, :len(D)] @ np.diag(np.maximum(D - lamda, 0)) @ V[:len(D), :]

    def loss(self, X):
        # \ell(X)=\|\sqrt{W} *(M-X)\|_F^2. Directly implement the formula.
        base_loss = np.linalg.norm(np.sqrt(self.W) * (self.M - X), 'fro') ** 2

        if self.relaxation:
            return 1/2 * base_loss + self.lamda * np.linalg.norm(X, 'nuc')
        else:
            return base_loss

    def _solver_baseline(self, X0, t=1):
        # Initialize variables.
        M, W, max_iter, tol = self.M, self.W, self.max_iter, self.tol
        # Iteration
        X = X0
        self.losses.append(self.loss(X))
        self.delta.append(1)
        for i in range(1, max_iter+1):
            Y = t * W * M + (1 - t * W) * X
            X = self.svd_threshold(Y, self.lamda) if self.relaxation else self.svd_k(Y, self.k)
            self.losses.append(self.loss(X))
            self.delta.append(np.abs((self.losses[-2] - self.losses[-1]) / self.losses[-2]))
            if self.delta[-1] < tol:
                print("Baseline solver converged at iteration " + str(i) + ".")
                break
            elif i == max_iter:
                print("Baseline solver reached the maximum number of iterations.")

        return X

    def _solver_nesterov(self, X0):
        # Initialize variables.
        M, W, max_iter, tol = self.M, self.W, self.max_iter, self.tol
        # Iteration
        X, X_prev = X0, X0
        self.losses.append(self.loss(X))
        self.delta.append(1)
        for i in range(1, max_iter+1):
            # V^{(i)}=X^{(i)}+\frac{i-1}{i+2}\left(X^{(i)}-X^{(i-1)}\right)
            V = X + (i - 1) / (i + 2) * (X - X_prev)
            # Y^{(i)}=t W * M+(1-t W) * V^{(i)}
            Y = W * M + (1 - W) * V
            # X^{(i+1)}=\operatorname{SVD}_k\left(Y^{(i)}\right)
            X_prev = X
            X = self.svd_threshold(Y, self.lamda) if self.relaxation else self.svd_k(Y, self.k)
            self.losses.append(self.loss(X))
            self.delta.append(np.abs((self.losses[-2] - self.losses[-1]) / self.losses[-2]))
            if self.delta[-1] < tol:
                print("Nesterov solver converged at iteration " + str(i) + ".")
                break
            elif i == max_iter:
                print("Nesterov solver reached the maximum number of iterations.")

        return X

    def _solver_anderson(self, X0, memory):
        # Initialization.
        M, W, max_iter, tol = self.M, self.W, self.max_iter, self.tol
        Y = W * M + (1 - W) * X0
        y = Y.flatten()
        X = self.svd_threshold(Y, self.lamda) if self.relaxation else self.svd_k(Y, self.k)
        self.losses.append(self.loss(X))
        self.delta.append(1)
        r_mat, f_mat = [], []
        # Iteration.
        for i in range(1, max_iter+1):
            f = (W*M + (1-W)*X).flatten()
            r_mat.append(f - y)
            f_mat.append(f)
            # solve OLS.
            r_mat_np = np.array(r_mat).T
            theta = np.linalg.inv(r_mat_np.T @ r_mat_np) @ np.ones(r_mat_np.shape[1])
            alpha = theta / np.sum(theta)
            # update.
            y = np.array(f_mat).T @ alpha
            Y = y.reshape(Y.shape)
            X = self.svd_threshold(Y, self.lamda) if self.relaxation else self.svd_k(Y, self.k)
            # drop the first column if the memory is full.
            if len(r_mat) > memory:
                r_mat.pop(0)
                f_mat.pop(0)
            # record loss.
            self.losses.append(self.loss(X))
            self.delta.append(np.abs((self.losses[-2] - self.losses[-1]) / self.losses[-2]))
            if self.delta[-1] < tol:
                print("Anderson solver converged at iteration " + str(i) + ".")
                break
            elif i == max_iter:
                print("Anderson solver reached the maximum number of iterations.")

        return X

    def solve(self, X0=None):
        # initialize X0.
        if X0 is None:
            X0 = np.zeros(self.M.shape)
        # solve the problem.
        if self.solver == 'baseline':
            return self._solver_baseline(X0)
        elif self.solver == 'nesterov':
            return self._solver_nesterov(X0)
        elif self.solver == 'anderson':
            return self._solver_anderson(X0, memory=self.memory)

    def plot_log_delta(self):
        # use seaborn style
        sns.set()
        # plot log delta
        plt.plot(np.log10(self.delta))
        plt.xlabel('Iteration')
        plt.ylabel(r'$\log(\Delta)$')
        plt.title("k = " + str(self.k))
        plt.axhline(y=np.log10(self.tol), color='k', linestyle='--')
        plt.show()

