from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad = self.calc_gradient(x, y)
        return self.update_weights(gradient=grad)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        L = x.shape[0]

        x_w = x @ self.w

        return 1 / L * (y - x_w).T @ (y - x_w)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        new_w = -self.lr() * gradient
        return new_w

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad = (-2 / x.shape[0]) * (y - x @ self.w).T @ x
        return grad


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # как создать два сэмпла подсказал гпт

        idxs = np.random.randint(0, x.shape[1], size=self.batch_size)

        idxs = np.unique(idxs)[:self.batch_size]

        x_sample = x[idxs]
        y_sample = y[idxs]

        return self.lr() * (-2 / self.batch_size) * (y_sample - x_sample @ self.w).T @ x_sample


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

        self.lr = 00.1

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient

        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (np.square(gradient))

        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)

        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = np.linalg.norm(self.w)

        return super().calc_gradient(x, y) + l2_gradient * self.mu / 2


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        vanilla_gradient = VanillaGradientDescent.calc_gradient(self, x, y)

        reg_gradient = BaseDescentReg.calc_gradient(self, x, y)

        combined_gradient = vanilla_gradient + reg_gradient

        return combined_gradient


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """

    def __init__(self, *args, mu=0, **kwargs):
        BaseDescentReg.__init__(self, *args, mu=mu, **kwargs)
        VanillaGradientDescent.__init__(self, *args, **kwargs)

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        return StochasticDescent.calc_gradient(self, x, y) + BaseDescentReg.calc_gradient(self, x, y)


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """

    def __init__(self, *args, mu=0, **kwargs):
        BaseDescentReg.__init__(self, *args, mu=mu, **kwargs)
        MomentumDescent.__init__(self, *args, **kwargs)

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        return MomentumDescent.calc_gradient(self, x, y) + BaseDescentReg.calc_gradient(self, x, y)


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

    def __init__(self, *args, mu=0, **kwargs):
        BaseDescentReg.__init__(self, *args, mu=mu, **kwargs)
        Adam.__init__(self, *args, **kwargs)

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        return Adam.calc_gradient(self, x, y) + BaseDescentReg.calc_gradient(self, x, y)


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
