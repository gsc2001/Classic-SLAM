from typing import Callable
# import jax.numpy as np
import numpy as np
import logging
print('hi')

class Optimizer:
    def __init__(self, f: Callable[[np.ndarray], np.ndarray],
                 jac: Callable[[np.ndarray], np.ndarray],
                 initial_values: np.ndarray, info_mat: np.ndarray, n_iter=10,
                 algo='GN', lm_lambda=1e-3):
        """
        Optmizer for given loss_function and jacobial function
        :param f:  Loss function
        :param jac: Function to compute jac at a state
        :param initial_values: Initial values of state
        :param algo: 'GN' for gauss newton, 'LM' for LM
        :param lm_lambda: Initial Lambda for LM

        """

        self.f = f
        self.jac = jac
        self.initial_values = initial_values
        self.info_mat = info_mat
        self.n_iter = n_iter
        self.algo = algo
        self.lm_lambda = lm_lambda
        self.prev_loss = -1

        self.F = lambda x: 0.5 * np.squeeze(f(x).T @ info_mat @ f(x))
        self.__current = self.initial_values.copy()
        self.__iter = 0

        self.losses = [self.loss()]
        print(f"INFO: Using ALGO: {algo}")
        print(f"INFO Initial Error: {self.F(self.__current)}")

    def get_current(self):
        return self.__current

    def get_gn_change(self) -> np.ndarray:
        J = self.jac(self.__current)
        H = J.T @ self.info_mat @ J
        b = J.T @ self.info_mat.T @ self.f(self.__current)
        dx = -np.linalg.inv(H) @ b
        return dx

    def get_lm_change(self) -> np.ndarray:
        J = self.jac(self.__current)
        H = J.T @ self.info_mat @ J
        H += self.lm_lambda * np.eye(H.shape[0])
        b = J.T @ self.info_mat.T @ self.f(self.__current)
        dx = -np.linalg.inv(H) @ b

        return dx
    
#     def get_lm_change(self) -> np.ndarray:
#         J = self.jac(self.__current)
#         H = J.T @ self.info_mat @ J
#         H += self.lm_lambda * jnp.eye(H.shape[0])
#         b = J.T @ self.info_mat.T @ self.f(self.__current)
#         dx = -jnp.linalg.pinv(H) @ b

#         return dx

    def print_info(self):
        print(f"Iteration: {self.__iter} / {self.n_iter}")
        print(f"Loss: {self.F(self.__current)}")
        # TODO: Draw plots

    def gn_update(self):
        self.__iter += 1
        dx = self.get_gn_change()
        self.__current += dx
        self.print_info()
        return abs(self.loss() - self.prev_loss) < 1e-4

    def lm_update(self):
        self.__iter += 1
        dx = self.get_lm_change()
        prev_loss = self.F(self.__current)
        new_current = self.__current + dx
        if self.F(new_current) < prev_loss:
            print("LM: Update Accepted")
            self.__current += dx
            self.lm_lambda /= 10
            self.print_info()
            return abs(self.loss() - self.prev_loss) < 1e-4
        else:
            print("LM: Update Rejected")
            self.__current += dx
            self.lm_lambda *= 10
            return False

    def loss(self):
        return self.F(self.__current)

    def optimize(self):
        self.print_info()
        for i in range(self.n_iter):
            to_break = False
            if self.algo == 'GN':
                to_break = self.gn_update()
            else:
                to_break = self.lm_update()
            if to_break:
                break
            self.losses.append(self.loss())
            self.prev_loss = self.loss()
