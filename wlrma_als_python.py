import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


class WLRMAwALS4Sparse:
    def __init__(self, M, W, k, lamda=0, max_iter=200, tol=1e-8, solver='baseline', memory=None):
        self.M = M
        self.W = W
        self.k = k
        self.lamda = lamda
        self.max_iter = max_iter
        self.tol = tol
        # specify the solver.
        if solver not in ['baseline', 'nesterov', 'anderson']:
            raise ValueError("Invalid solver. Allowed options are: 'baseline', 'nesterov', 'anderson'.")
        self.solver = solver
        if self.solver == 'anderson':
            self.memory = memory
        # record the loss, delta, solution rank and elapsed time.
        self.losses = []
        self.delta = []
        self.solution_rank = []
        self.elapsed_time = []

    def loss(self, A, B):
        diff = np.sqrt(self.W) * (self.M - A @ B.T)
        loss = np.linalg.norm(diff, 'fro') ** 2
        if self.lamda != 0:
            loss = (1/2 * loss
                    + (self.lamda / 2) * (np.linalg.norm(A, 'fro') ** 2 + np.linalg.norm(B, 'fro') ** 2))
        return loss

    def phi_operator(self, A, B):
        S = self.W * (self.M - A @ B.T)
        H_A = A @ np.linalg.inv(A.T @ A + self.lamda * np.eye(self.k))
        B_new = B @ A.T @ H_A + S.T @ H_A

        S = self.W * (self.M - A @ B_new.T)
        H_B = B_new @ np.linalg.inv(B_new.T @ B_new + self.lamda * np.eye(self.k))
        A_new = A @ B_new.T @ H_B + S @ H_B

        return A_new, B_new

    @staticmethod
    def calculate_solution_rank(A, B):
        # calculate the solution rank.
        U_A, D_A, V_A = np.linalg.svd(A)
        B_tilde = B @ V_A @ np.diag(D_A)
        U_B, D_B, V_B = np.linalg.svd(B_tilde)

        return np.sum(np.abs(D_B) > 1e-8)

    def _solver_baseline(self, A0, B0, verbose=False):
        start_time = time.time()
        # Initialization.
        A, B = A0, B0
        self.losses.append(self.loss(A, B))
        self.delta.append(1)
        self.solution_rank.append(self.calculate_solution_rank(A, B))
        self.elapsed_time.append(0)
        for i in range(1, self.max_iter+1):
            # update A and B alternatively.
            A, B = self.phi_operator(A, B)
            # record the loss, delta and solution rank.
            self.losses.append(self.loss(A, B))
            delta = np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-2]
            self.delta.append(delta)
            self.solution_rank.append(self.calculate_solution_rank(A, B))
            self.elapsed_time.append(time.time() - start_time)

            if verbose:
                # print the loss, delta and solution rank.
                print("Iteration " + str(i) + ": loss = " + str(self.losses[-1])
                      + ", delta = " + str(delta) + ", solution rank = " + str(self.solution_rank[-1]) + ".")

            # check the convergence.
            if delta < self.tol:
                print("WLRMAwALS4Sparse baseline solver converged at iteration " + str(i) + ".")
                break
            elif i == self.max_iter:
                print("WLRMAwALS4Sparse baseline solver reached the maximum number of iterations.")

        return A, B

    def _solver_nesterov(self, A0, B0, verbose=False):
        start_time = time.time()
        # Initialization.
        A, B = A0, B0
        A_prev, B_prev = A0, B0
        self.losses.append(self.loss(A, B))
        self.delta.append(1)
        self.solution_rank.append(self.calculate_solution_rank(A, B))
        self.elapsed_time.append(0)
        for i in range(1, self.max_iter + 1):
            # nestrov update.
            VA = A + (i - 1) / (i + 2) * (A - A_prev)
            VB = B + (i - 1) / (i + 2) * (B - B_prev)
            A_prev, B_prev = A, B
            # update A and B alternatively.
            A, B = self.phi_operator(VA, VB)
            # record the loss and check the convergence.
            self.losses.append(self.loss(A, B))
            delta = np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-2]
            self.delta.append(delta)
            self.solution_rank.append(self.calculate_solution_rank(A, B))
            self.elapsed_time.append(time.time() - start_time)

            if verbose:
                # print the loss, delta and solution rank.
                print("Iteration " + str(i) + ": loss = " + str(self.losses[-1])
                      + ", delta = " + str(delta) + ", solution rank = " + str(self.solution_rank[-1]) + ".")

            # check the convergence.
            if delta < self.tol:
                print("WLRMAwALS4Sparse nesterov solver converged at iteration " + str(i) + ".")
                break
            elif i == self.max_iter:
                print("WLRMAwALS4Sparse nesterov solver reached the maximum number of iterations.")

        return A, B

    def _solver_anderson(self, A0, B0, memory=3, verbose=False):
        start_time = time.time()
        # Initialization.
        A, B = A0, B0
        y = self.transform(A, B)
        self.losses.append(self.loss(A, B))
        self.delta.append(1)
        self.solution_rank.append(self.calculate_solution_rank(A, B))
        self.elapsed_time.append(0)
        r_mat, f_mat = [], []
        # Iteration.
        for i in range(1, self.max_iter+1):
            # get unaccelerated solution.
            A, B = self.phi_operator(A, B)
            loss = self.loss(A, B)
            # anderson acceleration.
            f = self.transform(A, B)
            r_mat.append(f - y)
            f_mat.append(f)
            # solve OLS.
            r_mat_np = np.array(r_mat).T
            theta = np.linalg.inv(r_mat_np.T @ r_mat_np) @ np.ones(r_mat_np.shape[1])
            alpha = theta / np.sum(theta)
            # get accelerated solution.
            y_acc = np.array(f_mat).T @ alpha
            A_acc, B_acc = self.inverse_transform(y_acc)
            loss_acc = self.loss(A_acc, B_acc)
            # compare the loss.
            if loss_acc < loss:
                A, B = A_acc, B_acc
                y = y_acc
                self.losses.append(loss_acc)
            else:
                y = self.transform(A, B)
                self.losses.append(loss)
            delta = np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-2]
            self.delta.append(delta)
            self.solution_rank.append(self.calculate_solution_rank(A, B))
            self.elapsed_time.append(time.time() - start_time)
            # drop the first column if the memory is full.
            if len(r_mat) > memory:
                r_mat.pop(0)
                f_mat.pop(0)

            if verbose:
                # print the loss, delta and solution rank.
                print("Iteration " + str(i) + ": loss = " + str(self.losses[-1])
                      + ", delta = " + str(self.delta[-1]) + ", solution rank = " + str(self.solution_rank[-1]) + ".")

            # check the convergence.
            if delta < self.tol:
                print("WLRMAwALS4Sparse anderson solver converged at iteration " + str(i) + ".")
                break
            elif i == self.max_iter:
                print("WLRMAwALS4Sparse anderson solver reached the maximum number of iterations.")

        return A, B

    @staticmethod
    def transform(A, B):
        return np.vstack([A, B]).flatten()

    def inverse_transform(self, y):
        n, p = self.M.shape
        mat = y.reshape((n + p, self.k))
        return mat[:n, :], mat[n:, :]

    def solve(self, A0=None, B0=None, verbose=False):
        # initialize A and B.
        if A0 is None:
            A0 = np.random.rand(self.M.shape[0], self.k)
        if B0 is None:
            B0 = np.random.rand(self.M.shape[1], self.k)
        # solve the problem.
        if self.solver == 'baseline':
            return self._solver_baseline(A0, B0, verbose=verbose)
        elif self.solver == 'nesterov':
            return self._solver_nesterov(A0, B0, verbose=verbose)
        elif self.solver == 'anderson':
            return self._solver_anderson(A0, B0, memory=self.memory, verbose=verbose)

    def plot_log_delta(self):
        sns.set()
        # plot log delta
        plt.plot(np.log10(self.delta))
        plt.xlabel('Iteration')
        plt.ylabel(r'$\log(\Delta)$')
        plt.title("k = " + str(self.k))
        plt.axhline(y=np.log10(self.tol), color='k', linestyle='--')
        plt.show()

    def plot_solution_rank(self):
        sns.set()
        # plot solution rank
        plt.plot(self.solution_rank)
        plt.xlabel('Iteration')
        plt.ylabel('Solution Rank')
        plt.title("k = " + str(self.k))
        plt.show()


def create_ratings_matrix(file_path):
    # Load the data
    ratings_data = pd.read_csv(file_path, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    # Pivot the table to create the matrix
    ratings_pivot = ratings_data.pivot(index='UserID', columns='MovieID', values='Rating')
    # Replace NaN values with 0 (indicating no rating)
    ratings_matrix = ratings_pivot.fillna(0).values

    return ratings_matrix
