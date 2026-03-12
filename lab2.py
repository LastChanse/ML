import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

'''
2. Регрессия с регуляризацией. Валидация параметров.
Используя данные из задания 1 построить регрессию с регуляризацией.

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

В качестве возможных базисных функций использовать полиномы, sin, cos, exp, sqrt (либо другие функции).
В качестве возможных значений коэффициента регуляризации задать ограниченный набор значений
(например lambda = {0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000}).
Для определения лучших параметров модели использовать итеративную рандомизированную процедуру (Монте-Карло) с разделением исходных данных на train, validation, test части .
Вычисление параметров модели производить на train части, валидацию параметров на validation части.
На лучших подобранных параметрах вычислить ошибку на test части. Значение ошибки на test части, 
лучшего коэффициента регуляризации и набор лучших базисных функций вывести на консоль.
В одном графическом окне построить график функции z(x) в виде непрерывной кривой,
t(x) в виде точек и график регрессии в виде непрерывной кривой, полученной на лучших параметрах модели.
'''

'''
F+ = (F^T*F)^-1 * F^T
||Fw-t||^2+a||w||^2
a||w||^2 - штраф
E(w)=1/2 (SUM<1,n>(ti-yi)^2+a*SUM<1,k>|wi|^2) - вместо 1/2 можно использовать 1/N ТОЛЬКО для вывода итогового результата
gradientE(w) = 0
w=F+*t
F+ с регуляризацией => F+ = (F^T*F+a*I^T)^-1 * F^T
I = единичная матрица

f1?
f2?
a=[0.0001,1e^-6,...,0.001]
f1=sin,cos,sin^2,...

x = 0,...,1 - всего 1000 шт.

Разбиваем x на 3 части
x 80% (случайных точек) - train
x 10% - test (Просто посчитать ошибку и вывести)
x 10% - validation (тут считаем ошибку E сравниваем с лучшей моделью если побеждаем лучшую обновляем лучшую модель)

можно брать индексы (НЕ x так как каждому x соответствует свой t) через шафлл

for i in range(1,100):
Посчитали 100 модели и получаем лучшую модель (w тоже сохраняем)
И смотрим на результат модели на test 

best_model = None
'''

def calculateY2(x, t, Fbase, lambdas, trainPercent, testPercent, validationPercent, Fbase_names):
    # Переменные
    N = len(x)
    best_l = None
    best_E = np.inf
    best_w = None
    best_F_test = None
    best_t_test = None

    # F(0)=1
    F = [np.ones_like(x)]

    for f in Fbase:
        F.append(f(x))

    F = np.array(F).T

    # Подбираем лучшие параметры модели по Монте-Карло
    for i in range(0, 100):
        # Перемешивание индексов
        index_list = np.random.permutation(N)

        train_end_idx = int(N * trainPercent)
        validation_end_idx = train_end_idx + int(N * validationPercent)
        test_end_idx = validation_end_idx + int(N * testPercent)

        train_idxs = index_list[:train_end_idx]
        validation_idxs = index_list[train_end_idx:validation_end_idx]
        test_idxs = index_list[validation_end_idx:test_end_idx]

        F_train, t_train = F[train_idxs], t[train_idxs]
        F_validation, t_validation = F[validation_idxs], t[validation_idxs]
        F_test, t_test = F[test_idxs], t[test_idxs]

        # Подбор регуляризации
        for l in lambdas:
            # Вычисление параметров модели производим на train части
            w_train = calculateW(F_train, t_train, l);

            # Валидацию параметров на validation части
            E = calculateE(F_validation, t_validation, w_train) # вычисляем ошибку

            # Выбираем лучшие параметры для test части
            if E < best_E:
                best_E = E
                best_w = w_train
                best_l = l
                best_F_test = F_test
                best_t_test = t_test

    # На лучших подобранных параметрах вычисляем ошибку на test части.
    best_test_E = calculateE(best_F_test, best_t_test, best_w)
    '''
    Значение ошибки на test части, 
    лучшего коэффициента регуляризации и
    набор лучших базисных функций вывести на консоль.
    '''
    print("Ошибка тест части E=", best_test_E)
    print("Лучший коэффициент регуляризации l=:", best_l)
    # Ищем список лучших функций
    best_functions = ["1"]

    for i in range(len(Fbase_names)):
        # Проверяем наличие вклада (чем выше параметр тем значительнее вклад функции)
        if abs(best_w[i + 1]) > 1e-6: # i+1 так как Fbase_names без F(0)='1', а best_w с 1
            best_functions.append(Fbase_names[i])

    print("Набор лучших базисных функций:", best_functions)

    # Возвращаем регрессию для графика
    Y = F @ best_w
    return Y

# Вычисление параметров
def calculateW(F,t,l):
    # Единичная матрица размерности тойже что и F
    I = np.eye(F.shape[1])
    F_plus = np.linalg.pinv(F.T @ F + l * I) @ F.T
    w = F_plus @ t
    return w

# Вычисление ошибки
def calculateE(F,t,w):
    Y = F @ w
    error = np.mean((t - Y) ** 2)
    return error

def HW2():
    N = 1000  # кол-во элементов в обучающей выборке
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(N)
    t = z + error

    # Базисные функции
    # Полиномы
    Xstart, Xend = 1,11
    X = [lambda x, i=i: x ** i for i in range(Xstart, Xend)] # x,x^2,...x^10
    X_names = [f"x^{i}" for i in range(Xstart, Xend)]

    Fbase = X+[np.cos, np.sin, np.exp, np.sqrt]
    Fbase_names = ["cos", "sin", "exp", "sqrt"] + X_names

    lambdas = {0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000}

    trainPercent, testPercent, validationPercent = 0.8, 0.1, 0.1

    Y = calculateY2(x, t, Fbase, lambdas, trainPercent, testPercent, validationPercent, Fbase_names)
    '''
    В одном графическом окне построить график функции z(x) в виде непрерывной кривой,
    t(x) в виде точек и график регрессии в виде непрерывной кривой, полученной на лучших параметрах модели.
    '''
    my_plot1 = plt  # axes[0, 0]

    my_plot1.plot(x, z, 'g-', linewidth=2, label='z(x)')
    my_plot1.scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    my_plot1.plot(x, Y, 'r-', linewidth=2, label=f'best Y')
    my_plot1.legend()
    my_plot1.grid(True)

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    print("start")
    HW2()
    print("done")