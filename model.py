import numpy as np


class SVM:
    def __init__(self, learning_rate, lambda_param, n_iters) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y >= 0, 1, -1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(self.weights, x_i) - self.bias) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.weights
                    self.weights -= self.lr * dw
                else:
                    dw = 2 * self.lambda_param * self.weights - y_[idx] * x_i
                    db = y_[idx]
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db

    def predict(self, X):
        output = np.dot(self.weights, X) - self.bias
        return np.sign(output)

