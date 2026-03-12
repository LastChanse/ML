import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

'''
1. Регрессия
Построить регрессию используя аналитическое решение через псевдообратную матрицу Мура-Пенроза. Данные задать следующим образом:

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

В качестве базисных функций использовать полиномы до степени M.

- В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 1
- В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 8
- В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 100
- Построить график зависимости ошибки E(w) от степени полинома M. M меняется от 1 до 100.
'''

'''
Y = X * w
w = F+ * Y

F+ = inv(transpose(F) * F) * transpose(F) # Псевдо обратная матрица
E(w)=sum((ti-yi)^2) / N
error = np.sum((t - Y) ** 2) / N
    
'''

def demo():
    X = np.array([1,3,9])
    Y = np.array([2,4,5])
    M = 3

    # Матрица плана
    F = [
        [1, 1, 1],
        [1, 3, 9],
        [1, 9, 81],
    ]
    # Примеры из обучающей выборки
    #t = F*w
    F_plus = np.linalg.inv(X.T*X)*X.T
    w = F_plus * Y
    N = 1000
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)

def calculateY(x,M,t,Fbase):
    F = []
    for i in range(0, M + 1):
        F.append(np.pow(x, i))
    F = np.array(F).T

    F_plus = np.linalg.inv(F.T @ F) @ F.T
    w = F_plus @ t
    Y = F @ w
    return Y

def HW1():
    N = 1000 # кол-во элементов в обучающей выборке
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(N)
    t = z + error

    # M - макс степень полинома
    fig, axes = plt.subplots(2, 2)

    # 1 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 1
    M = 1
    Y1 = calculateY(x,M,t)

    axes[0, 0].plot(x, z, 'g-', linewidth=2, label='z(x)')
    axes[0, 0].scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    axes[0, 0].plot(x, Y1, 'r--', linewidth=2, label=f'M=1')
    axes[0, 0].set_xlabel('Значения x')
    axes[0, 0].set_ylabel('Значения функций')
    axes[0, 0].set_title(f'M = 1')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 8
    M = 8
    Y2 = calculateY(x,M,t)

    axes[0, 1].plot(x, z, 'g-', linewidth=2, label='z(x)')
    axes[0, 1].scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    axes[0, 1].plot(x, Y2, 'r--', linewidth=2, label=f'M=8')
    axes[0, 1].set_xlabel('Значения x')
    axes[0, 1].set_ylabel('Значения функций')
    axes[0, 1].set_title(f'M = 8')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 100
    M = 100
    Y3 = calculateY(x,M,t)

    axes[1, 0].plot(x, z, 'g-', linewidth=2, label='z(x)')
    axes[1, 0].scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    axes[1, 0].plot(x, Y3, 'r--', linewidth=2, label=f'M=100')
    axes[1, 0].set_xlabel('Значения x')
    axes[1, 0].set_ylabel('Значения функций')
    axes[1, 0].set_title(f'M = 100')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4 Построить график зависимости ошибки E(w) от степени полинома M. M меняется от 1 до 100.
    # sum((ti-yi)^2) / N # средне квадратичная ошибка (Y)
    # 1..M # Степени полинома (X)
    E = []
    m = [i for i in range(1,101)]
    for M in m:
        Y = calculateY(x, M, t)
        E.append(np.sum((t - Y) ** 2) / N)

    axes[1, 1].plot(m, E, 'r--', linewidth=2, label=f'E(w)')
    axes[1, 1].set_xlabel('Степень полинома M')
    axes[1, 1].set_ylabel('Ошибка')
    axes[1, 1].set_title('Зависимость ошибки от M')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.show(block=True)

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
Для определения лучших параметров модели
использовать итеративную рандомизированную процедуру (Монте-Карло) с разделением исходных данных на train, validation, test части .
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
'''

def demo2():
    N = 1000 # кол-во элементов в обучающей выборке
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(N)
    t = z + error

    Fbase = [math.cos,math.sin,math.exp,math.sqrt]

    # M - макс степень полинома
    M = 1
    # fig, axes = plt.subplots(2, 2)
    Y1 = calculateY2(x,M,t,Fbase)

    my_plot1 = plt#axes[0, 0]

    my_plot1.plot(x, z, 'g-', linewidth=2, label='z(x)')
    my_plot1.scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    my_plot1.plot(x, Y1, 'r--', linewidth=2, label=f'M=1')
    # my_plot1.set_xlabel('Значения x')
    # my_plot1.set_ylabel('Значения функций')
    # my_plot1.set_title(f'M = 1')
    my_plot1.legend()
    my_plot1.grid(True)

    plt.tight_layout()
    plt.show(block=True)
    print('test')

def calculateY2(x,M,t,Fbase):
    F = []
    for i in range(0, M + 1):
        F.append(np.pow(Fbase(x), i))
    F = np.array(F).T
    
    #F + = (F ^ T * F + a * I ^ T) ^ -1 * F ^ T
    F_plus = np.linalg.inv(F.T @ F) @ F.T
    w = F_plus @ t
    Y = F @ w
    return Y

def HW2():
    N = 1000 # кол-во элементов в обучающей выборке
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(N)
    t = z + error

    # M - макс степень полинома
    # fig, axes = plt.subplots(2, 2)

    # 1 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 1
    M = 1
    Y1 = calculateY(x,M,t)

    my_plot1 = plt#axes[0, 0]

    my_plot1.plot(x, z, 'g-', linewidth=2, label='z(x)')
    my_plot1.scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    my_plot1.plot(x, Y1, 'r--', linewidth=2, label=f'M=1')
    # my_plot1.set_xlabel('Значения x')
    # my_plot1.set_ylabel('Значения функций')
    # my_plot1.set_title(f'M = 1')
    my_plot1.legend()
    my_plot1.grid(True)

    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    print("start")
    # HW1()
    # demo()
    demo2()
    # HW2()
    print("done")