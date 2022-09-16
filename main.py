from model import SVM
import numpy as np
import matplotlib.pyplot as plt


def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


def visualize_svm(X, w, b):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=X[:, 2])

    x0_1 = np.amin(X[:, 0])
    x0_2 = get_hyperplane_value(x0_1, w, b, 0)
    nsv1 = get_hyperplane_value(x0_1, w, b, -1)
    psv1 = get_hyperplane_value(x0_1, w, b, 1)

    x1_1 = np.amax(X[:, 0])
    x1_2 = get_hyperplane_value(x1_1, w, b, 0)
    nsv2 = get_hyperplane_value(x1_1, w, b, -1)
    psv2 = get_hyperplane_value(x1_1, w, b, 1)

    ax.plot([x0_1, x1_1], [x0_2, x1_2], 'y--')
    ax.plot([x0_1, x1_1], [nsv1, nsv2], 'k')
    ax.plot([x0_1, x1_1], [psv1, psv2], 'k')
    plt.show()


def main():
    dataset = np.array([[1, 3, 1],
                        [2, 4, 1],
                        [3, 4, 1],
                        [1, 5, 1],
                        [3, 6, 1],
                        [2, 1, -1],
                        [3, 0, -1],
                        [4, 1, -1],
                        [4, 2, -1],
                        [5, 3, -1]])

    model = SVM(lr=0.01, lambda_param=0.001, n_iters=1000)
    model.fit(dataset[:, [0, 1]], dataset[:, [2]])
    visualize_svm(dataset, model.weights, model.bias)


if __name__ == "__main__":
    main()
