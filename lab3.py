from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# Стандартизация
def standardize(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


# Полиномиальные признаки
def polynomial_features(X, degree):
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    return X_poly


# Функция ошибки
# def compute_loss(F, t, w, alpha):
#     return 0.5 * np.sum((t - F @ w) ** 2) + (alpha / 2) * np.sum(w ** 2)
def compute_loss(F, t, w, alpha):
    return (1/len(F)) * np.sum((t - F @ w) ** 2)


# Градиентный спуск
def gradient_descent(F, t, alpha, lr=1e-5, max_iter=1000):
    N, D = F.shape

    w = np.random.normal(0, 0.1, size=D)

    losses = []

    for i in range(max_iter):
        # градиент по формуле:
        # -F^T t + F^T F w + alpha w
        grad = -(F.T @ t) + (F.T @ F) @ w + alpha * w

        w = w - lr * grad

        loss = compute_loss(F, t, w, alpha)
        losses.append(loss)

    return w, losses


# Основная функция
def HW3():
    data = load_boston()
    X = data.data
    t = data.target

    N = len(X)

    # перемешивание
    idx = np.random.permutation(N)

    train_end = int(0.8 * N)
    val_end = int(0.9 * N)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    degrees = [1, 2, 3]
    alphas = [0]

    best_E = np.inf
    best_model = None

    for degree in degrees:
        # признаки
        F = polynomial_features(X, degree)

        # стандартизация
        F = standardize(F)
        t = standardize(t)

        # добавляем F0=[1,1,1,1,1...]
        F = np.c_[np.ones(len(F)), F]

        F_train = F[train_idx]
        t_train = t[train_idx]

        F_val = F[val_idx]
        t_val = t[val_idx]

        F_test = F[test_idx]
        t_test = t[test_idx]

        for alpha in alphas:
            w, losses = gradient_descent(F_train, t_train, alpha)

            E_val = compute_loss(F_val, t_val, w, alpha)

            if E_val < best_E:
                best_E = E_val
                best_model = (w, degree, alpha, F_test, t_test, losses)

    w, degree, alpha, F_test, t_test, losses = best_model

    test_error = compute_loss(F_test, t_test, w, alpha)

    print("\n=== ЛУЧШАЯ МОДЕЛЬ ===")
    print("Степень полинома:", degree)
    print("alpha:", alpha)
    print("Ошибка (val):", best_E)
    print("Ошибка (test):", test_error)

    # график
    plt.plot(losses)
    plt.xlabel("Итерация")
    plt.ylabel("Ошибка")
    plt.title("Сходимость GD (теоретическая форма)")
    plt.show()


if __name__ == "__main__":
    HW3()