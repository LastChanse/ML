from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

LAMBDA = 0.1

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std


def compute_loss(X, y, w, l):
    N = len(y)
    y_pred = X @ w
    return (1 / (2 * N)) * np.sum((y - y_pred) ** 2) + (l / 2) * np.sum(w ** 2)


def gradient_descent(X, y, l, lr=0.01, max_iter=1000, tol_grad=1e-6, tol_w=1e-6):
    N, D = X.shape

    # случайная инициализация весов из нормального распределения
    w = np.random.normal(0, 0.1, size=D)

    losses = []

    for i in range(max_iter):
        y_pred = X @ w
        grad = -(1 / N) * (X.T @ (y - y_pred)) + l * w

        w_new = w - lr * grad

        # критерии остановки
        if np.linalg.norm(grad) < tol_grad:
            print(f"Остановка по норме градиента на итерации {i}")
            break

        if np.linalg.norm(w_new - w) < tol_w:
            print(f"Остановка по изменению весов на итерации {i}")
            break

        w = w_new

        loss = compute_loss(X, y, w, l)
        losses.append(loss)

    return w, losses


def HW3():
    np.random.seed(42)

    boston = load_boston()
    X = boston.data
    y = boston.target

    # стандартизация данных
    X, mean, std = standardize(X)

    # добавляем bias
    X = np.c_[np.ones(len(X)), X]

    N = len(X)
    trainPercent, testPercent, validationPercent = 0.8, 0.1, 0.1

    index_list = np.random.permutation(N)

    train_end = int(0.8 * N)
    val_end = int(0.9 * N)

    train_idxs = index_list[:train_end]
    validation_idxs = index_list[train_end:val_end]
    test_idxs = index_list[val_end:]
    degrees = [1, 2, 3]
    lambdas = [0, 0.01, 0.1, 1]

    best_model = None
    best_params = None
    best_E = np.inf

    for degree in degrees:
        # создаём новые признаки
        X_poly = polynomial_features(boston.data, degree)

        # стандартизация
        X_poly, mean, std = standardize(X_poly)

        # добавляем bias
        X_poly = np.c_[np.ones(len(X_poly)), X_poly]

        # разбиение
        X_train = X_poly[train_idxs]
        y_train = y[train_idxs]

        X_val = X_poly[validation_idxs]
        y_val = y[validation_idxs]

        X_test_local = X_poly[test_idxs]
        y_test_local = y[test_idxs]

        losses = []
        for l in lambdas:
            # шаг learning rate ~0.01
            w, losses = gradient_descent(X_train, y_train, l, lr=0.01)

            E_val = compute_loss(X_val, y_val, w, l)

            if E_val < best_E:
                best_E = E_val
                best_model = (w, degree, l)
                best_params = (degree, l)
                best_test_data = (X_test_local, y_test_local)
    w, degree, l = best_model
    X_test_best, y_test_best = best_test_data

    test_error = compute_loss(X_test_best, y_test_best, w, l)

    print("\n=== ЛУЧШАЯ МОДЕЛЬ ===")
    print("Степень полинома:", degree)
    print("Lambda:", l)
    print("Ошибка на обучающей выборке:", best_E)
    print("Ошибка на тестовой выборке:", test_error)

    # график ошибки
    plt.plot(losses)
    plt.xlabel("Итерация")
    plt.ylabel("Ошибка")
    plt.title("Сходимость градиентного спуска")
    plt.show()

    return w

# Генерация полиномиальных признаков
def polynomial_features(X, degree):
    X_poly = X.copy()

    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))

    return X_poly

if __name__ == "__main__":
    print("start")
    HW3()
    print("done")