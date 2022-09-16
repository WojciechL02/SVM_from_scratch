import numpy as np


class SVM:
    def __init__(self, learning_rate, lambda_param, n_iters) -> None:
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self):
        pass

    def predict(self, X):
        output = np.dot(self.weights, X) - self.bias
        return np.sign(output)





